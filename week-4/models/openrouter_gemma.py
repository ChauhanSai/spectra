from __future__ import annotations

import base64
import io
import os
import sys

from .common import extract_label

PROMPT = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

DEFAULT_MODEL = "google/gemma-3n-e2b-it:free"

_client = None
_first_error_logged = False


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


def predict_label(pil_img) -> str:
    global _first_error_logged
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        if not _first_error_logged:
            _first_error_logged = True
            print("OpenRouter Gemma: set OPENROUTER_API_KEY in env/.env", file=sys.stderr)
        return "unknown"
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        client = _get_client()
        model = (os.environ.get("OPENROUTER_GEMMA_MODEL") or DEFAULT_MODEL).strip()
        resp = client.chat.completions.create(
            model=model,
            max_tokens=64,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                PROMPT
                                + " Now, based only on the MRI image, classify this scan."
                            ),
                        },
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
        err = str(e).lower()
        if "401" in err or "unauthorized" in err:
            print("\nOpenRouter Gemma: unauthorized (check OPENROUTER_API_KEY)", file=sys.stderr)
            sys.exit(1)
        if not _first_error_logged:
            _first_error_logged = True
            print("OpenRouter Gemma API error (first occurrence):", e, file=sys.stderr)
        return "unknown"

