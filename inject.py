from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

POSITIONS = ("top_left", "top_right", "center", "bottom", "bottom_left", "bottom_right")


def inject_text(
    image: Image.Image,
    text: str,
    *,
    position: str = "top_left",
    font_size: int = 24,
    opacity: float = 0.35,
) -> Image.Image:
    if position not in POSITIONS:
        position = "top_left"

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _get_font(font_size)
    tw, th = _text_size(draw, text, font)
    margin = 12
    w, h = base.size

    x, y = _position_xy(position, w, h, tw, th, margin)
    alpha = int(max(0.0, min(1.0, opacity)) * 255)
    draw.text((x, y), text, fill=(255, 255, 255, alpha), font=font)

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
