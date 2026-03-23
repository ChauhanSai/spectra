import os
import random
import cv2
from config.settings import COCO_TRAIN_DIR, SAMPLE_LIST_PATH, NUM_IMAGES, RANDOM_SEED

MIN_SIDE = 256
MIN_STD = 20
TARGET_TOTAL = 200
LEGACY_SAMPLE_LIST_PATH = "outputs/sample/sampled_images.txt"

def passes_filter(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    if h < MIN_SIDE or w < MIN_SIDE:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.std() < MIN_STD:
        return False

    return True


def resolve_sample_list_path():
    if os.path.exists(LEGACY_SAMPLE_LIST_PATH):
        return LEGACY_SAMPLE_LIST_PATH
    return SAMPLE_LIST_PATH


def load_existing_samples(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        rows = [line.strip() for line in f if line.strip()]
    return list(dict.fromkeys(rows))

def main():
    random.seed(RANDOM_SEED)
    sample_list_path = resolve_sample_list_path()
    existing_samples = load_existing_samples(sample_list_path)
    existing_set = set(existing_samples)

    target_total = min(NUM_IMAGES, TARGET_TOTAL)
    needed = max(0, target_total - len(existing_samples))

    all_images = [
        f for f in os.listdir(COCO_TRAIN_DIR)
        if f.lower().endswith(".jpg")
    ]

    print(f"Found {len(all_images)} images total.")
    print(
        f"Existing sampled images: {len(existing_samples)} | "
        f"Need to add: {needed}"
    )

    random.shuffle(all_images)

    new_samples = []
    for img in all_images:
        if len(new_samples) >= needed:
            break
        if img in existing_set:
            continue

        img_path = os.path.join(COCO_TRAIN_DIR, img)
        if passes_filter(img_path):
            new_samples.append(img)
            existing_set.add(img)

    sampled_images = existing_samples + new_samples
    print(
        f"Added {len(new_samples)} new images after filtering. "
        f"Total now: {len(sampled_images)}"
    )

    os.makedirs(os.path.dirname(sample_list_path), exist_ok=True)

    with open(sample_list_path, "w") as f:
        for img in sampled_images:
            f.write(img + "\n")

    print(f"Saved {len(sampled_images)} image names to {sample_list_path}")

if __name__ == "__main__":
    main()
