from __future__ import annotations

import base64
import io
import os
import sys

from .common import extract_label

PROMPT = "Classify this brain MRI image. Respond with ONLY ONE of these labels: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."

MODEL_URL = "https://clarifai.com/openbmb/miniCPM/models/MiniCPM-V-2_6-vllm"

_model = None
_first_error_logged = False


def _get_model():
    global _model
    if _model is not None:
        return _model
    from clarifai.client import Model
    _model = Model(url=MODEL_URL)
    return _model


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        from clarifai.client.input import Inputs
    except ImportError:
        if not _first_error_logged:
            _first_error_logged = True
            print("Clarifai backend: pip install -U clarifai", file=sys.stderr)
        return "unknown"

    pat = (os.environ.get("CLARIFAI_PAT") or "").strip()
    if not pat:
        if not _first_error_logged:
            _first_error_logged = True
            print("Clarifai: set CLARIFAI_PAT in .env", file=sys.stderr)
        return "unknown"

    try:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64}"

        model = _get_model()
        input_data = Inputs.get_multimodal_input(
            input_id="",
            image_url=image_url,
            raw_text=PROMPT,
        )
        results = model.predict(
            [input_data],
            inference_params=dict(temperature=0, max_tokens=64),
        )
        text = ""
        outputs = getattr(results, "outputs", None)
        if outputs is not None and len(outputs) > 0:
            out = outputs[0]
            data = getattr(out, "data", None)
            if data is not None:
                txt = getattr(data, "text", None)
                if txt is not None:
                    text = (getattr(txt, "raw", None) or str(txt) or "").strip()
        if not text:
            raise ValueError("Empty response")
        return extract_label(text)
    except Exception as e:
        err_str = str(e)
        if "restricted to dedicated compute" in err_str or "MODEL_PREDICTION_FAILED" in err_str:
            print("\nClarifai: MiniCPM-V-2_6 requires dedicated compute; your request used shared nodepool.", file=sys.stderr)
            print("Upgrade to dedicated compute in Clarifai, or use another model (e.g. moondream, llama4-scout).", file=sys.stderr)
            sys.exit(1)
        if not _first_error_logged:
            _first_error_logged = True
            print("Clarifai MiniCPM API error (first occurrence):", e, file=sys.stderr)
        return "unknown"
