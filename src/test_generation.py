import os
import cv2
from config.settings import (
    COCO_TRAIN_DIR,
    INJECTION_STYLES,
    TEXT_SIZES
)
from utils.saliency import get_salient_cells
from utils.inject_text import draw_text_in_cell

TEST_IMAGE = "000000000025.jpg"
TEST_WORD = "flower"

OUTPUT_DIR = "outputs/test"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_path = os.path.join(COCO_TRAIN_DIR, TEST_IMAGE)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Test image not found")

    most_cell, least_cell = get_salient_cells(img_path)

    placements = [
        ("most", most_cell),
        ("least", least_cell),
    ]

    for placement_name, (row, col) in placements:
        for style_name, template in INJECTION_STYLES.items():
            text = template.format(word=TEST_WORD)

            for size_name in TEXT_SIZES.keys():
                for contrast_name in ["low", "high"]:
                    img_copy = img.copy()

                    attacked = draw_text_in_cell(
                        img_copy,
                        text,
                        row, col,
                        size_name=size_name,
                        contrast_level=f"{contrast_name}_contrast"
                    )

                    fname = f"{placement_name}_{style_name}_{size_name}_{contrast_name}.jpg"
                    out_path = os.path.join(OUTPUT_DIR, fname)
                    cv2.imwrite(out_path, attacked)

                    print("Saved", fname)

    print("Done test generation.")


if __name__ == "__main__":
    main()
