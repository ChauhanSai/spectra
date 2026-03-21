import os
import random
import cv2
from config.settings import COCO_TRAIN_DIR, SAMPLE_LIST_PATH, NUM_IMAGES, RANDOM_SEED

MIN_SIDE = 256
MIN_STD = 20

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

def main():
    random.seed(RANDOM_SEED)

    all_images = [
        f for f in os.listdir(COCO_TRAIN_DIR)
        if f.lower().endswith(".jpg")
    ]

    print(f"Found {len(all_images)} images total.")

    random.shuffle(all_images)

    sampled_images = []
    for img in all_images:
        if len(sampled_images) >= NUM_IMAGES:
            break

        img_path = os.path.join(COCO_TRAIN_DIR, img)
        if passes_filter(img_path):
            sampled_images.append(img)

    print(f"Kept {len(sampled_images)} images after filtering.")

    os.makedirs(os.path.dirname(SAMPLE_LIST_PATH), exist_ok=True)

    with open(SAMPLE_LIST_PATH, "w") as f:
        for img in sampled_images:
            f.write(img + "\n")

    print(f"Saved {len(sampled_images)} image names to {SAMPLE_LIST_PATH}")

if __name__ == "__main__":
    main()
