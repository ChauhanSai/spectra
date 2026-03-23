from __future__ import annotations

import os
import sys
import time
import tempfile
import subprocess

from .common import extract_label

OLLAMA_MODEL = os.environ.get("OLLAMA_LLAMA32_VISION_MODEL", "llama3.2-vision")

_first_error_logged = False


def _restart_ollama():
    """Kill and restart the Ollama server."""
    print("  [Ollama] Restarting server...", file=sys.stderr, flush=True)
    try:
        subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(3)
    env = os.environ.copy()
    env["OLLAMA_KEEP_ALIVE"] = "-1"
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    time.sleep(10)  # give it plenty of time to start
    print("  [Ollama] Server restarted", file=sys.stderr, flush=True)


def _call_ollama_robust(abs_path: str, prompt: str) -> str:
    """Call Ollama with retry + auto-restart on failure."""
    import ollama as ollama_lib

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama_lib.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt, "images": [abs_path]}],
            )
            text = ""
            if response and "message" in response and response["message"]:
                content = response["message"].get("content")
                if isinstance(content, str):
                    text = content.strip()
            if not text:
                raise ValueError("Empty response from Ollama")
            return extract_label(text)
        except Exception as e:
            print(f"  [Ollama] Attempt {attempt+1}/{max_retries} failed: {e}",
                  file=sys.stderr, flush=True)
            if attempt < max_retries - 1:
                _restart_ollama()
            else:
                raise
    return "unknown"


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
        return _call_ollama_robust(abs_path, prompt)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print(f"Llama 3.2 Vision error: {e}", file=sys.stderr)
        return "unknown"
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def predict_label_with_policy(pil_img, policy_prompt: str) -> str:
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
            policy_prompt + "\n\n"
            "Classify this brain MRI image. Respond with ONLY ONE of these labels and nothing else: "
            "glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."
        )
        return _call_ollama_robust(abs_path, prompt)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print(f"Llama 3.2 Vision (policy) error: {e}", file=sys.stderr)
        return "unknown"
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass
