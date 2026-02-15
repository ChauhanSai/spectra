import os
import re
import sys
from google import genai
from google.genai import types

CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_INSTRUCTIONS = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

_first_error_logged = False


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


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace(" ", "_")
    return text


def _extract_label(text: str) -> str | None:
    normalized = _normalize(text)
    for class_name in CLASSES:
        if class_name in normalized:
            return class_name
    for class_name in CLASSES:
        if class_name.replace("_", "") in normalized.replace("_", ""):
            return class_name
    return None


def predict_label(pil_img) -> str:
    global _first_error_logged
    import io

    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        prompt = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                temperature=0.0,
            ),
        )

        text = _get_response_text(resp)
        if not text:
            raise ValueError("Empty or non-text response")

        label = _extract_label(text)
        if label is not None:
            return label
        raise ValueError("No valid label in response")
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("Gemini API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
