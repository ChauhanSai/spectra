from __future__ import annotations

import base64
import io
import os
import sys

from .common import extract_label

PROMPT = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

DEFAULT_MODEL = "google/gemma-3-4b-it"

_client = None
_first_error_logged = False


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        token = (os.environ.get("HF_TOKEN") or "").strip()
        _client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
        )
    return _client


def predict_label(pil_img) -> str:
    global _first_error_logged
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        if not _first_error_logged:
            _first_error_logged = True
            print("Gemma HF backend: set HF_TOKEN in .env", file=sys.stderr)
        return "unknown"
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        client = _get_client()
        model = os.environ.get("HF_GEMMA_MODEL", DEFAULT_MODEL)
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
        err = str(e).lower()
        if "model_not_supported" in err or "not supported by any provider" in err:
            print(
                "\nHF router: 'google/gemma-3-4b-it' is not available for your account/providers.",
                file=sys.stderr,
            )
            print(
                "Options: (1) create a paid Inference Endpoint for Gemma 3 on HF, "
                "(2) use OpenRouter's gemma-3-4b-it:free API, or (3) run Gemma locally with transformers.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not _first_error_logged:
            _first_error_logged = True
            print("HF Gemma API error (first occurrence):", e, file=sys.stderr)
        return "unknown"

