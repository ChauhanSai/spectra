from __future__ import annotations

import base64
import io
import os
import sys

from .common import extract_label

PROMPT = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

DEFAULT_MODEL = "opengvlab/internvl3-2b:free"

_client = None
_first_error_logged = False


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    return _client


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        client = _get_client()
        model = os.environ.get("OPENROUTER_INTERNVL_MODEL", DEFAULT_MODEL)
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
        if "401" in err_str or "invalid api key" in err_str or "unauthorized" in err_str:
            print("\nOpenRouter: invalid or missing API key. Get a free key at https://openrouter.ai/settings/keys", file=sys.stderr)
            sys.exit(1)
        if "402" in err_str or "insufficient" in err_str or "credit" in err_str:
            print("\nOpenRouter: insufficient credit.", file=sys.stderr)
            sys.exit(1)
        if "404" in err_str or "no endpoints found" in err_str:
            print("\nOpenRouter: InternVL3-2B free endpoint not available. Try Llama 4 Scout (HF) or Qwen (local).", file=sys.stderr)
            sys.exit(1)
        if not _first_error_logged:
            _first_error_logged = True
            print("OpenRouter InternVL3.2B API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
