import base64
import io
import os
import sys

from .common import extract_label

PROMPT = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

_client = None
_first_error_logged = False


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        client = _get_client()
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_VISION_MODEL", "gpt-4o"),
            max_tokens=64,
            messages=[
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this brain MRI image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
        )
        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        if not text:
            raise ValueError("Empty response")
        return extract_label(text)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("OpenAI API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
