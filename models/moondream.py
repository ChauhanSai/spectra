from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
from urllib.error import HTTPError

from .common import extract_label

QUESTION = "Classify this brain MRI image. Respond with ONLY ONE of these labels: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."

BASE_URL = "https://api.moondream.ai/v1"
_first_error_logged = False


def predict_label(pil_img) -> str:
    global _first_error_logged
    api_key = (os.environ.get("MOONDREAM_API_KEY") or "").strip()
    if not api_key:
        if not _first_error_logged:
            _first_error_logged = True
            print("Moondream: set MOONDREAM_API_KEY in .env", file=sys.stderr)
        return "unknown"
    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        payload = json.dumps({
            "image_url": f"data:image/jpeg;base64,{b64}",
            "question": QUESTION,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{BASE_URL}/query",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Moondream-Auth": api_key,
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("answer") or "").strip()
        if not text:
            raise ValueError("Empty response")
        return extract_label(text)
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if e.code == 403:
            print("\nMoondream API: 403 Forbidden.", file=sys.stderr)
            if body:
                print("Response:", body[:500], file=sys.stderr)
            print("Check: MOONDREAM_API_KEY in .env, key from https://console.moondream.ai", file=sys.stderr)
            sys.exit(1)
        if not _first_error_logged:
            _first_error_logged = True
            print("Moondream API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("Moondream API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
