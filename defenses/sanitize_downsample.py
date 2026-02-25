from __future__ import annotations

import io

from PIL import Image, ImageFilter


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def sanitize_image(
    img: Image.Image,
    max_dim: int = 512,
    blur_radius: float = 1.2,
    jpeg_quality: int = 70,
) -> Image.Image:
    rgb = _ensure_rgb(img)
    w, h = rgb.size
    scale = 1.0
    if max(w, h) > max_dim and max_dim > 0:
        scale = float(max_dim) / float(max(w, h))
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        rgb = rgb.resize(new_size, Image.BILINEAR)
    if blur_radius and blur_radius > 0:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    buf.seek(0)
    out = Image.open(buf)
    return _ensure_rgb(out)

