from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

POSITIONS = ("top_left", "top_right", "center", "bottom", "bottom_left", "bottom_right")
SALIENT_POSITIONS = ("most_salient", "least_salient")
CONTRAST_LEVELS = ("low_contrast", "medium_contrast", "high_contrast")
LOW_CONTRAST_OFFSET = 35


def _clamp(x: float) -> int:
    return int(max(0, min(255, x)))


def _pick_text_color(bg_rgb: tuple[int, int, int], contrast_level: str | None) -> tuple[int, int, int]:
    if contrast_level not in CONTRAST_LEVELS:
        return (255, 255, 255)
    r, g, b = bg_rgb
    if contrast_level == "medium_contrast":
        brightness = (r + g + b) / 3
        return (0, 0, 0) if brightness > 127 else (255, 255, 255)
    if contrast_level == "low_contrast":
        brightness = (r + g + b) / 3
        if brightness > 127:
            return (
                _clamp(r - LOW_CONTRAST_OFFSET),
                _clamp(g - LOW_CONTRAST_OFFSET),
                _clamp(b - LOW_CONTRAST_OFFSET),
            )
        return (
            _clamp(r + LOW_CONTRAST_OFFSET),
            _clamp(g + LOW_CONTRAST_OFFSET),
            _clamp(b + LOW_CONTRAST_OFFSET),
        )
    spread = max(r, g, b) - min(r, g, b)
    if spread < 20:
        brightness = (r + g + b) / 3
        return (0, 0, 0) if brightness > 127 else (255, 255, 255)
    if g >= r and g >= b:
        return (0, 0, 255)
    if r >= g and r >= b:
        return (255, 255, 0)
    return (0, 255, 255)


def _sample_region_rgb(pixels: np.ndarray, x: int, y: int, tw: int, th: int, w: int, h: int) -> tuple[int, int, int]:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + max(tw, 1))
    y2 = min(h, y + max(th, 1))
    region = pixels[y1:y2, x1:x2]
    if region.size == 0:
        return (255, 255, 255)
    r, g, b = region.reshape(-1, 3).mean(axis=0)
    return _clamp(r), _clamp(g), _clamp(b)


def inject_text(
    image: Image.Image,
    text: str,
    *,
    position: str = "top_left",
    font_size: int = 24,
    opacity: float = 0.35,
    contrast: str | None = None,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _get_font(font_size)
    tw, th = _text_size(draw, text, font)
    margin = 12
    w, h = base.size

    if position in SALIENT_POSITIONS:
        img_rgb = np.array(base.convert("RGB"))
        from saliency import get_salient_cells, GRID_SIZE
        most_cell, least_cell = get_salient_cells(img_rgb)
        row, col = most_cell if position == "most_salient" else least_cell
        cell_w = w // GRID_SIZE
        cell_h = h // GRID_SIZE
        x1 = col * cell_w
        y1 = row * cell_h
        x = x1 + max(margin, (cell_w - tw) // 2)
        y = y1 + max(margin, (cell_h - th) // 2)
    else:
        pos = position if position in POSITIONS else "top_left"
        x, y = _position_xy(pos, w, h, tw, th, margin)

    if contrast in CONTRAST_LEVELS:
        pixels = np.array(base.convert("RGB"))
        bg_rgb = _sample_region_rgb(pixels, x, y, tw, th, w, h)
        r, g, b = _pick_text_color(bg_rgb, contrast)
        alpha = 255 if contrast == "high_contrast" else int(max(0.0, min(1.0, opacity)) * 255)  # medium/low use opacity
        fill = (r, g, b, alpha)
    else:
        alpha = int(max(0.0, min(1.0, opacity)) * 255)
        fill = (255, 255, 255, alpha)

    draw.text((x, y), text, fill=fill, font=font)

    return Image.alpha_composite(base, overlay).convert("RGB")


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ("Arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def _position_xy(
    position: str,
    img_w: int,
    img_h: int,
    tw: int,
    th: int,
    margin: int,
) -> tuple[int, int]:
    if position == "top_left":
        return margin, margin
    if position == "top_right":
        return max(margin, img_w - tw - margin), margin
    if position == "center":
        return max(margin, (img_w - tw) // 2), max(margin, (img_h - th) // 2)
    if position == "bottom":
        return max(margin, (img_w - tw) // 2), max(margin, img_h - th - margin)
    if position == "bottom_left":
        return margin, max(margin, img_h - th - margin)
    if position == "bottom_right":
        return max(margin, img_w - tw - margin), max(margin, img_h - th - margin)
    return margin, margin
