from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
from PIL import Image


PredictFn = Callable[[Image.Image], str]


def majority_vote(preds: list[str]) -> str:
    if not preds:
        return "unknown"
    counts = Counter(preds)
    return counts.most_common(1)[0][0]


def apply_color_variants(img: Image.Image) -> list[Image.Image]:
    rgb = img.convert("RGB")
    arr = np.array(rgb)

    gray = np.mean(arr, axis=2).astype(np.uint8)
    gray_3ch = np.stack([gray, gray, gray], axis=2)

    red = arr.copy()
    red[:, :, 1] = 0
    red[:, :, 2] = 0

    green = arr.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0

    blue = arr.copy()
    blue[:, :, 0] = 0
    blue[:, :, 1] = 0

    return [
        rgb,
        Image.fromarray(gray_3ch, mode="RGB"),
        Image.fromarray(red, mode="RGB"),
        Image.fromarray(green, mode="RGB"),
        Image.fromarray(blue, mode="RGB"),
    ]


def predict_with_color_vote(img: Image.Image, predict_fn: PredictFn) -> tuple[str, list[str]]:
    variants = apply_color_variants(img)
    preds = [predict_fn(variant) for variant in variants]
    return majority_vote(preds), preds
