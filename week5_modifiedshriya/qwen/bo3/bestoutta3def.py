from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter


PredictFn = Callable[[Image.Image], str]


def majority_vote(preds: list[str]) -> str:
    if not preds:
        return "unknown"
    counts = Counter(preds)
    return counts.most_common(1)[0][0]


def apply_transformations(img: Image.Image) -> list[Image.Image]:
    rgb = img.convert("RGB")

    blur = rgb.filter(ImageFilter.GaussianBlur(radius=1.5))

    arr = np.array(rgb).astype(np.int16)
    noise = arr + np.random.normal(scale=25, size=arr.shape).astype(np.int16)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    noise_img = Image.fromarray(noise, mode="RGB")

    rot = rgb.rotate(15, resample=Image.BILINEAR)

    return [blur, noise_img, rot]


def predict_with_bo3(img: Image.Image, predict_fn: PredictFn) -> tuple[str, list[str]]:
    variants = apply_transformations(img)
    preds = [predict_fn(variant) for variant in variants]
    return majority_vote(preds), preds
