from __future__ import annotations

import base64
import io
import os
import sys

from .common import extract_label

PROMPT = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

DEFAULT_MODEL = "OpenGVLab/InternVL3-38B-Instruct"

_client = None
_first_error_logged = False


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_TOKEN"),
        )
    return _client


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        client = _get_client()
        model = os.environ.get("HF_INTERNVL_MODEL", DEFAULT_MODEL)
        resp = client.chat.completions.create(
            model=model,
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
        err_str = str(e).lower()
        if "not supported" in err_str or "model_not_supported" in err_str:
            print("\nInternVL3-38B is not available on the HF router for your account.", file=sys.stderr)
            print("Enable a provider in HF settings or run the model locally (transformers/LMDeploy).", file=sys.stderr)
            sys.exit(1)
        if not _first_error_logged:
            _first_error_logged = True
            print("HF InternVL3 API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
