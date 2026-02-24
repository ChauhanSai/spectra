import io
import os
import sys
from google import genai
from google.genai import types

from .common import extract_label

MODEL_NAME = "gemini-2.5-flash"
PROMPT = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

_client = None
_first_error_logged = False


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client


def _get_response_text(resp) -> str:
    try:
        return (resp.text or "").strip()
    except ValueError:
        pass
    try:
        candidates = getattr(resp, "candidates", None) or []
        for c in candidates:
            content = getattr(c, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for p in parts:
                if hasattr(p, "text") and p.text:
                    return (p.text or "").strip()
    except Exception:
        pass
    return ""


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        client = _get_client()
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                system_instruction=PROMPT,
                temperature=0.0,
            ),
        )
        text = _get_response_text(resp)
        if not text:
            raise ValueError("Empty or non-text response")
        return extract_label(text)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("Gemini API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
