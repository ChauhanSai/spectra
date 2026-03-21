import cv2
import numpy as np
import sys
import os

"""Color channel voting defense module.

This implements a 5-way ensemble defense that processes images through
different color representations: RGB, grayscale, red-only, green-only,
and blue-only channels. The model's predictions on all five are then
combined via majority vote.
"""


def apply_color_variants(img):
    """Return list of 5 color-transformed variants of the image.
    
    Returns:
        [original RGB, grayscale (3ch), red-only, green-only, blue-only]
    """
    # 1. Original RGB
    rgb = img.copy()

    # 2. Grayscale (replicate to 3 channels for model compatibility)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 3. Red channel only (keep R, zero B and G)
    red = img.copy()
    red[:, :, 0] = 0  # Zero out blue
    red[:, :, 1] = 0  # Zero out green

    # 4. Green channel only (keep G, zero B and R)
    green = img.copy()
    green[:, :, 0] = 0  # Zero out blue
    green[:, :, 2] = 0  # Zero out red

    # 5. Blue channel only (keep B, zero R and G)
    blue = img.copy()
    blue[:, :, 1] = 0  # Zero out green
    blue[:, :, 2] = 0  # Zero out red

    return [rgb, gray_3ch, red, green, blue]


def majority_vote(preds):
    """Return the most common string in preds."""
    if not preds:
        return "unknown"
    counts = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python colorvote.py path/to/image.jpg")
        sys.exit(1)

    inp = sys.argv[1]
    img = cv2.imread(inp)
    if img is None:
        print(f"couldn't open {inp}")
        sys.exit(1)

    variants = apply_color_variants(img)

    base = os.path.splitext(os.path.basename(inp))[0]
    names = [
        f"{base}_rgb.jpg",
        f"{base}_gray.jpg",
        f"{base}_red.jpg",
        f"{base}_green.jpg",
        f"{base}_blue.jpg"
    ]

    # Write to current working directory (project root)
    for name, var in zip(names, variants):
        outpath = os.path.join(".", name)
        cv2.imwrite(outpath, var)
        print(f"wrote {outpath}")
