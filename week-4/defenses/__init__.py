from __future__ import annotations

from typing import Callable, Optional

from PIL import Image

from . import security_policy


DefenseFn = Callable[[Image.Image, str], Image.Image]


def _noop(img: Image.Image, mode: str) -> Image.Image:
    return img


def get_defense(name: Optional[str]) -> DefenseFn:
    nid = (name or "").strip().lower()
    if nid in ("security_policy", "policy", "camel"):
        return security_policy.apply_security_policy
    return _noop


def get_policy_prompt(name: Optional[str]) -> str | None:
    """Return the hardened system prompt if a policy defense is active."""
    nid = (name or "").strip().lower()
    if nid in ("security_policy", "policy", "camel"):
        return security_policy.get_policy_prompt()
    return None
