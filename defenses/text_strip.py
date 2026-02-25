from __future__ import annotations

import io
from typing import Sequence

from PIL import Image, ImageFilter


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def strip_text(
    img: Image.Image,
    blur_radius: float = 2.0,
    min_confidence: int = 50,
    languages: str | None = None,
) -> Image.Image:
    try:
        import pytesseract
        from pytesseract import Output
    except Exception:
        return img
    rgb = _ensure_rgb(img)
    try:
        data = pytesseract.image_to_data(
            rgb,
            lang=languages or "eng",
            output_type=Output.DICT,
        )
    except Exception:
        return img
    n = len(data.get("text", []))
    if n == 0:
        return rgb
    out = rgb.copy()
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf_str = str(data.get("conf", ["-1"])[i])
        try:
            conf = int(float(conf_str))
        except ValueError:
            conf = -1
        if not txt or conf < min_confidence:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        if w <= 0 or h <= 0:
            continue
        box = (max(0, x), max(0, y), x + w, y + h)
        region = out.crop(box)
        blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        out.paste(blurred, box)
    return out

