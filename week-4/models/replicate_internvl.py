from __future__ import annotations

import os
import sys
import tempfile

from .common import extract_label

PROMPT = """You are a medical image classifier. Respond with ONLY ONE of the following labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

MODEL_VERSION = "lucataco/internvl3_5-30b:92713bb56e2e4827477501fc512d80d7287bc34e4ccb7c306afd9bdaef8b7eeb"

_first_error_logged = False


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        import replicate
    except ImportError:
        if not _first_error_logged:
            _first_error_logged = True
            print("Replicate backend: pip install replicate", file=sys.stderr)
        return "unknown"

    if not os.environ.get("REPLICATE_API_TOKEN"):
        if not _first_error_logged:
            _first_error_logged = True
            print("Replicate backend: set REPLICATE_API_TOKEN in .env", file=sys.stderr)
        return "unknown"

    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_img.save(f, format="PNG")
            path = f.name
        with open(path, "rb") as f:
            output = replicate.run(
                MODEL_VERSION,
                input={
                    "images": f,
                    "prompt": "Classify this brain MRI image. Respond with ONLY ONE of these labels: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor.",
                    "max_new_tokens": 64,
                    "temperature": 0,
                },
            )
        text = (output or "").strip() if isinstance(output, str) else ""
        if not text:
            raise ValueError("Empty response")
        return extract_label(text)
    except Exception as e:
        err_str = str(e).lower()
        if "402" in err_str or "insufficient credit" in err_str:
            print("\nReplicate: insufficient credit. Add billing at https://replicate.com/account/billing", file=sys.stderr)
            sys.exit(1)
        if not _first_error_logged:
            _first_error_logged = True
            print("Replicate InternVL3.5 API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass
