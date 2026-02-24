from __future__ import annotations

import os
import sys
import tempfile

from .common import extract_label

MODEL_ID = "OpenGVLab/InternVL3-2B-hf"

_pipe = None
_first_error_logged = False


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("pip install transformers torch")
    print(f"Loading {MODEL_ID} (local pipeline)...", file=sys.stderr)
    _pipe = pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        trust_remote_code=True,
    )
    return _pipe


def predict_label(pil_img) -> str:
    global _first_error_logged
    path = None
    try:
        pipe = _get_pipe()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_img.save(f, format="PNG")
            path = f.name
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": path if os.path.isabs(path) else os.path.abspath(path)},
                    {"type": "text", "text": "Classify this brain MRI. Respond with ONLY ONE of these labels: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."},
                ],
            },
        ]
        out = pipe(messages, max_new_tokens=64, return_full_text=False)
        text = ""
        if out and len(out) > 0:
            msg = out[0]
            if isinstance(msg, dict) and "generated_text" in msg:
                text = (msg["generated_text"] or "").strip()
            elif isinstance(msg, dict) and "message" in msg:
                content = msg["message"].get("content", "")
                text = (content or "").strip()
            elif isinstance(msg, str):
                text = msg.strip()
        if not text:
            raise ValueError("Empty response")
        return extract_label(text)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("Local InternVL3-2B error (first occurrence):", e, file=sys.stderr)
        return "unknown"
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass
