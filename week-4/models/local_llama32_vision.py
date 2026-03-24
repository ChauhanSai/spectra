from __future__ import annotations

import os
import sys

from .common import extract_label

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

_model = None
_processor = None
_first_error_logged = False


def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    import torch
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    device = _get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL_ID} on {device}...", file=sys.stderr)
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    _model.eval()
    return _model, _processor


PROMPT = (
    "Classify this brain MRI image. Respond with ONLY ONE of these labels "
    "and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."
)


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        import torch
        model, processor = _load_model()
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=pil_img, text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return extract_label(response)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print(f"Llama 3.2 Vision error: {e}", file=sys.stderr)
        return "unknown"


def predict_label_with_policy(pil_img, policy_prompt: str) -> str:
    global _first_error_logged
    try:
        import torch
        model, processor = _load_model()
        full_prompt = policy_prompt + "\n\n" + PROMPT
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": full_prompt},
            ]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=pil_img, text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return extract_label(response)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print(f"Llama 3.2 Vision (policy) error: {e}", file=sys.stderr)
        return "unknown"
