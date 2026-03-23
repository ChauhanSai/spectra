import cv2
import numpy as np
import sys
import os


def majority_vote(preds):
    """Return the most common string in preds."""
    if not preds:
        return "unknown"
    counts = {}
    for p in preds:
        counts[p] = counts.get(p, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


def apply_transformations(img):
    """Return a list of three variants: blur, noise, and rotated."""
    h, w = img.shape[:2]

    # 1) Gaussian blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 2) additive Gaussian noise
    noise = img.astype(np.int16)
    noise += np.random.normal(scale=25, size=noise.shape).astype(np.int16)
    noise = np.clip(noise, 0, 255).astype(np.uint8)

    # 3) rotation by 15 degrees around center
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
    rot = cv2.warpAffine(img, M, (w, h))

    return [blur, noise, rot]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python bestoutta3def.py path/to/image.jpg")
        sys.exit(1)

    inp = sys.argv[1]
    img = cv2.imread(inp)
    if img is None:
        print(f"couldn't open {inp}")
        sys.exit(1)

    variants = apply_transformations(img)

    base = os.path.splitext(os.path.basename(inp))[0]
    out_dir = os.path.dirname(inp) or "."
    names = [f"{base}_blur.jpg", f"{base}_noise.jpg", f"{base}_rot.jpg"]

    for name, var in zip(names, variants):
        outpath = os.path.join(out_dir, name)
        cv2.imwrite(outpath, var)
        print(f"wrote {outpath}")
