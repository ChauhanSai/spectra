from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PIL import Image


@dataclass
class OcrFirewallResult:
    blocked: bool
    reasons: List[str]
    text: str


IMPERATIVE_KEYWORDS = [
    "ignore",
    "follow",
    "execute",
    "obey",
    "override",
    "prepend",
    "append",
]

SYSTEM_OVERRIDE_PATTERNS = [
    "you are",
    "system prompt",
    "system message",
    "as system",
    "as the system",
    "instruction",
    "jailbreak",
]

REASONING_PATTERNS = [
    "step by step",
    "first,",
    "second,",
    "third,",
    "let us think",
    "let's think",
]


def analyze_text(
    img: Image.Image,
    languages: str | None = None,
) -> OcrFirewallResult:
    try:
        import pytesseract
    except Exception:
        return OcrFirewallResult(False, [], "")
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    try:
        text = pytesseract.image_to_string(
            rgb,
            lang=languages or "eng",
        )
    except Exception:
        return OcrFirewallResult(False, [], "")
    lower = (text or "").lower()
    reasons: List[str] = []
    for kw in IMPERATIVE_KEYWORDS:
        if kw in lower:
            reasons.append(f"imperative:{kw}")
    for pat in SYSTEM_OVERRIDE_PATTERNS:
        if pat in lower:
            reasons.append(f"system_override:{pat}")
    for pat in REASONING_PATTERNS:
        if pat in lower:
            reasons.append(f"reasoning:{pat}")
    blocked = bool(reasons)
    return OcrFirewallResult(blocked=blocked, reasons=reasons, text=text or "")


def should_block(img: Image.Image, languages: str | None = None) -> bool:
    return analyze_text(img, languages=languages).blocked

