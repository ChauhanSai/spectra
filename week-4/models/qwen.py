from __future__ import annotations

import sys
from .common import extract_label

PROMPT = """You are a medical image classifier. This is a brain MRI. Respond with ONLY ONE of these labels and nothing else: glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor."""

_model = None
_processor = None
_first_error_logged = False


def _get_device():
    import platform
    try:
        import torch
        if platform.system() == "Darwin":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    device = _get_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"Loading {model_id} on {device}...", file=sys.stderr)
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        _model = _model.to("cpu")
    _model.eval()
    return _model, _processor


def predict_label(pil_img) -> str:
    global _first_error_logged
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError as e:
        print("Qwen backend: missing dependency.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("Install with: pip install transformers qwen-vl-utils torch torchvision accelerate", file=sys.stderr)
        raise
    try:
        model, processor = _load_model()
        device = next(model.parameters()).device
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        print("  (Qwen: generating on CPU, may take 1–3 min)...", flush=True, file=sys.stderr)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        label = extract_label(response)
        if label == "unknown" and response and not _first_error_logged:
            _first_error_logged = True
            print("Qwen raw response (first unknown):", repr(response[:200]), file=sys.stderr)
        return label
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("Qwen API error (first occurrence):", e, file=sys.stderr)
        return "unknown"


def predict_label_with_policy(pil_img, policy_prompt: str) -> str:
    global _first_error_logged
    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        print("Qwen backend: missing dependency.", file=sys.stderr)
        raise
    try:
        model, processor = _load_model()
        device = next(model.parameters()).device
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": policy_prompt + "\n\nClassify this brain MRI image."},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        return extract_label(response)
    except Exception as e:
        if not _first_error_logged:
            _first_error_logged = True
            print("Qwen (policy) error:", e, file=sys.stderr)
        return "unknown"
