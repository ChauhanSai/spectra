from __future__ import annotations

import os
import sys
import tempfile

from .common import extract_label

OLLAMA_MODEL = os.environ.get("OLLAMA_LLAMA32_VISION_MODEL", "llama3.2-vision")

_first_error_logged = False


def predict_label(pil_img) -> str:
    global _first_error_logged
    path = None
    try:
        import ollama
    except ImportError:
        raise ImportError("pip install ollama")
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_img.save(f, format="PNG")
            path = f.name
        abs_path = os.path.abspath(path)
        prompt = (
            "Classify this brain MRI image. Respond with ONLY ONE of these labels and nothing else: "
            "glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."
        )
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [abs_path],
                }
            ],
        )
        text = ""
        if response and "message" in response and response["message"]:
            content = response["message"].get("content")
            if isinstance(content, str):
                text = content.strip()
        if not text:
            raise ValueError("Empty response")
        return extract_label(text)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print(
                f"Llama 3.2 Vision (Ollama) error (first occurrence): {e}",
                file=sys.stderr,
            )
        return "unknown"
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass
