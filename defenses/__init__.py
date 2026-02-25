from __future__ import annotations

from typing import Callable, Optional

from PIL import Image

from . import text_strip
from . import sanitize_downsample
from . import ocr_firewall


DefenseFn = Callable[[Image.Image, str], Image.Image]


def _noop(img: Image.Image, mode: str) -> Image.Image:
    return img


def get_defense(name: Optional[str]) -> DefenseFn:
    nid = (name or "").strip().lower()
    if nid in ("strip_text", "text_strip", "ocr_strip"):
        def f(img: Image.Image, mode: str) -> Image.Image:
            return text_strip.strip_text(img)
        return f
    if nid in ("sanitize", "downsample", "sanitize_downsample"):
        def f(img: Image.Image, mode: str) -> Image.Image:
            return sanitize_downsample.sanitize_image(img)
        return f
    return _noop


def should_block_by_firewall(img: Image.Image) -> bool:
    return ocr_firewall.should_block(img)

