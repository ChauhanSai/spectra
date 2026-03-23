import os
import glob
import re
import torch
import cv2
import tempfile
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from defense.colorvote import apply_color_variants, majority_vote

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28
MAX_NEW_TOKENS = 12
PROMPT = "What is the main object in this image? Reply with one word only."
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

RESULTS_DIR = "outputs/results"
FLUSH_EVERY = 50


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device):
    print(f"Loading model {MODEL_ID} on {device} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model.eval()
    return model, processor


def infer_single(image_path, model, processor, device):
    # read original image once
    img = cv2.imread(image_path)
    if img is None:
        return "unknown"

    # generate 5 color variants
    variants = apply_color_variants(img)
    preds = []

    # run inference on each variant and collect predictions
    for var in variants:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, var)
            raw = infer_single_raw(tmp.name, PROMPT, model, processor, device)
        preds.append(coerce_to_one_word(raw))

    # return majority vote across all 5 variants
    return majority_vote(preds)


def coerce_to_one_word(text):
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("\n", " ").replace("|", " ").replace("/", " ")

    cleaned = re.sub(
        r"^\s*the\s+main\s+object\s+in\s+(this\s+)?image\s*(is|:)?\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(r"^\s*main\s+object\s*(is|:)?\s*", "", cleaned)

    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", cleaned)

    if len(tokens) == 1:
        return tokens[0]

    stopwords = {
        "the", "main", "object", "in", "this", "image", "is",
        "a", "an", "of", "there", "appears", "to", "be"
    }
    filtered = [tok for tok in tokens if tok not in stopwords]
    if filtered:
        return filtered[0]
    if tokens:
        return tokens[-1]
    return "unknown"


def infer_single_raw(image_path, prompt, model, processor, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
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

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0].strip()


def parse_condition(images_dir):
    parts = images_dir.split(os.sep)
    return parts[-2]


def run_inference_for_condition(images_dir, model, processor, device):
    word = parse_condition(images_dir)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_name = f"{word}results.txt"
    out_path = os.path.join(RESULTS_DIR, out_name)

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not image_files:
        return

    print(f"Processing {len(image_files)} images -> {out_name}")

    with open(out_path, "w") as f:
        for idx, img_path in enumerate(tqdm(image_files, desc=out_name), start=1):
            fname = os.path.basename(img_path)
            try:
                result = infer_single(img_path, model, processor, device)
            except Exception as e:
                result = f"ERROR: {e}"
            f.write(f"{fname} | {result}\n")
            if idx % FLUSH_EVERY == 0:
                f.flush()
        f.flush()


def find_all_image_dirs(base="outputs/attacked"):
    pattern = os.path.join(base, "*", "images")
    return sorted(glob.glob(pattern))


def main():
    device = get_device()
    print(f"Using device: {device}")

    model, processor = load_model(device)

    image_dirs = find_all_image_dirs()
    print(f"Found {len(image_dirs)} condition directories.")

    for images_dir in image_dirs:
        run_inference_for_condition(images_dir, model, processor, device)

    print("Done.")


if __name__ == "__main__":
    main()