from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.common import extract_label

MODEL_ID = os.environ.get("LLAMA32_VISION_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")

_pipe = None
_first_error_logged = False


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise ImportError("pip install -U transformers torch accelerate sentencepiece") from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this runner, but no GPU runtime was detected. "
            "In Colab, switch to Runtime -> Change runtime type -> GPU."
        )

    print(f"Loading {MODEL_ID} on cuda...", file=sys.stderr)
    _pipe = pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
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
        abs_path = path if os.path.isabs(path) else os.path.abspath(path)

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a medical image classifier. Respond with ONLY ONE of these labels "
                            "and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": abs_path},
                    {"type": "text", "text": "Classify this brain MRI image."},
                ],
            },
        ]

        out = pipe(messages, max_new_tokens=64)
        text = ""
        if out and len(out) > 0:
            msg = out[0]
            if isinstance(msg, dict) and "generated_text" in msg:
                gen = msg["generated_text"]
                if isinstance(gen, list) and gen:
                    last = gen[-1]
                    if isinstance(last, dict) and "content" in last:
                        text = (last["content"] or "").strip()
                elif isinstance(gen, str):
                    text = gen.strip()
            elif isinstance(msg, str):
                text = msg.strip()
        if not text:
            raise ValueError("Empty response")
        return extract_label(text)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("Llama 3.2 local error (first occurrence):", e, file=sys.stderr)
        return "unknown"
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass
