from __future__ import annotations

import logging
import os
import sys
import warnings

from .common import extract_label

MODEL_ID = os.environ.get("MINICPM_V26_MODEL", "openbmb/MiniCPM-V-2_6")

_model = None
_tokenizer = None
_first_error_logged = False


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(f"pip install -U transformers torch — {e!r}") from e
    _warning_once_seen: set = set()

    def _safe_warning_once(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[1], type) and issubclass(args[1], Warning):
            key = (args[0], args[1])
            if key not in _warning_once_seen:
                _warning_once_seen.add(key)
                warnings.warn(args[0], args[1], stacklevel=2)
            return
        self.warning(*args, **kwargs)

    logging.Logger.warning_once = _safe_warning_once
    try:
        import transformers.logging as tf_logging
        tf_logging.set_verbosity(tf_logging.ERROR)
    except Exception:
        pass
    get_class_from_dynamic_module = None
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except Exception:
        pass
    print(f"Loading {MODEL_ID} (local MiniCPM-V-2.6)...", file=sys.stderr)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
    if class_ref and get_class_from_dynamic_module is not None:
        try:
            _cls = get_class_from_dynamic_module(class_ref, MODEL_ID, trust_remote_code=True)
            if not hasattr(_cls, "all_tied_weights_keys"):
                _cls.all_tied_weights_keys = {}
        except Exception:
            pass
    _model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    if not hasattr(_model, "all_tied_weights_keys"):
        _model.all_tied_weights_keys = {}
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return _model, _tokenizer


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        model, tokenizer = _get_model_and_tokenizer()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        prompt = (
            "Classify this brain MRI image. Respond with ONLY ONE of these labels and nothing else: "
            "glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."
        )
        msgs = [{"role": "user", "content": [pil_img, prompt]}]
        res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        text = (res or "").strip()
        if not text:
            raise ValueError("Empty response")
        return extract_label(text)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("Local MiniCPM-V-2.6 error (first occurrence):", e, file=sys.stderr)
        return "unknown"
