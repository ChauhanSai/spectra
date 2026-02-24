from __future__ import annotations

import base64
import io
import os
import sys

from .common import extract_label

PROMPT = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

BASE_URL = "https://api.clarifai.com/v2/ext/openai/v1"
DEFAULT_MODEL = "https://clarifai.com/mistralai/completion/models/Ministral-3-14B-Reasoning-2512/versions/9e8a48eb1b6e412ebd661cc84b96c156"

_client = None
_first_error_logged = False


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = (os.environ.get("CLARIFAI_PAT") or os.environ.get("CLARIFAI_API_KEY") or "").strip()
        _client = OpenAI(base_url=BASE_URL, api_key=api_key)
    return _client


def predict_label(pil_img) -> str:
    global _first_error_logged
    api_key = (os.environ.get("CLARIFAI_PAT") or os.environ.get("CLARIFAI_API_KEY") or "").strip()
    if not api_key:
        if not _first_error_logged:
            _first_error_logged = True
            print("Clarifai Ministral: set CLARIFAI_PAT or CLARIFAI_API_KEY in .env", file=sys.stderr)
        return "unknown"
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        client = _get_client()
        model = (os.environ.get("CLARIFAI_MINISTRAL_MODEL") or DEFAULT_MODEL).strip()
        resp = client.chat.completions.create(
            model=model,
            max_tokens=64,
            temperature=0,
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
            print("Clarifai Ministral API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
