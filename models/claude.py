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
        from anthropic import Anthropic
        _client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        client = _get_client()
        msg = client.messages.create(
            model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        )
        text = (msg.content[0].text or "").strip() if msg.content else ""
        if not text:
            raise ValueError("Empty response")
        return extract_label(text)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("Claude API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
