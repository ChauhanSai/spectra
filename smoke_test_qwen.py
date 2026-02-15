import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TEST_IMAGE = "outputs/test/most_direct_large_high.jpg"
PROMPT = "What is the main object in this image? Answer with ONE word."

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

dtype = torch.float16 if device == "mps" else torch.float32

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
).to(device)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 256,
    max_pixels=512 * 512,
)

img_path = os.path.abspath(TEST_IMAGE)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{img_path}"},
            {"type": "text", "text": PROMPT},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    out_ids = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
    )

gen_trim = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
out_text = processor.batch_decode(
    gen_trim, skip_special_tokens=True
)[0].strip()

print("model output:", out_text)
