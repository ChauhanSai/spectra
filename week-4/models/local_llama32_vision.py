from __future__ import annotations

import os
import sys
import time
import tempfile
import subprocess

from .common import extract_label

OLLAMA_MODEL = os.environ.get("OLLAMA_LLAMA32_VISION_MODEL", "llama3.2-vision")

_first_error_logged = False
_request_count = 0
_RESTART_EVERY = 20  # restart Ollama every N requests to prevent OOM


def _ensure_ollama_running():
    """Check if Ollama is responsive; restart if not."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        pass
    # Restart Ollama
    print("  [Ollama] Restarting server...", file=sys.stderr, flush=True)
    try:
        subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(2)
    env = os.environ.copy()
    env["OLLAMA_KEEP_ALIVE"] = "-1"
    subprocess.Popen(
        ["nohup", "ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        preexec_fn=os.setpgrp,
    )
    time.sleep(8)
    # Verify it came back
    try:
        import ollama
        ollama.list()
        print("  [Ollama] Server restarted OK", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"  [Ollama] Restart failed: {e}", file=sys.stderr, flush=True)
        return False


def _maybe_restart():
    """Periodically restart Ollama to avoid GPU memory buildup."""
    global _request_count
    _request_count += 1
    if _request_count % _RESTART_EVERY == 0:
        print(f"  [Ollama] Preventive restart after {_request_count} requests...",
              file=sys.stderr, flush=True)
        try:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=5)
        except Exception:
            pass
        time.sleep(3)
        _ensure_ollama_running()


def _call_ollama(abs_path: str, prompt: str, max_retries: int = 3) -> str:
    """Call Ollama with auto-restart on failure."""
    import ollama as ollama_lib

    _maybe_restart()

    for attempt in range(max_retries):
        try:
            _ensure_ollama_running()
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
                raise ValueError("Empty response")
            return extract_label(text)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [Ollama] Attempt {attempt+1} failed: {e}. Restarting...",
                      file=sys.stderr, flush=True)
                try:
                    subprocess.run(["pkill", "-f", "ollama"], capture_output=True, timeout=5)
                except Exception:
                    pass
                time.sleep(5)
                _ensure_ollama_running()
            else:
                raise
    return "unknown"


def predict_label(pil_img) -> str:
    global _first_error_logged
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_img.save(f, format="PNG")
            path = f.name
        abs_path = os.path.abspath(path)
        prompt = (
            "Classify this brain MRI image. Respond with ONLY ONE of these labels and nothing else: "
            "glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."
        )
        return _call_ollama(abs_path, prompt)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print(f"Llama 3.2 Vision (Ollama) error (first occurrence): {e}", file=sys.stderr)
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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_img.save(f, format="PNG")
            path = f.name
        abs_path = os.path.abspath(path)
        prompt = (
            policy_prompt + "\n\n"
            "Classify this brain MRI image. Respond with ONLY ONE of these labels and nothing else: "
            "glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."
        )
        return _call_ollama(abs_path, prompt)
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
