from __future__ import annotations

from typing import Callable, Optional

from PIL import Image

from . import security_policy
from . import defensive_token
from . import prompt_separation
from . import combined_defense


DefenseFn = Callable[[Image.Image, str], Image.Image]


def _noop(img: Image.Image, mode: str) -> Image.Image:
    return img


def get_defense(name: Optional[str]) -> DefenseFn:
    nid = (name or "").strip().lower()
    if nid in ("security_policy", "policy", "camel"):
        return security_policy.apply_security_policy
    if nid in ("defensive_token", "defense_token", "defense_tokens"):
        return defensive_token.apply_defensive_token
    if nid in ("prompt_separation", "struq", "structured_query"):
        return prompt_separation.apply_prompt_separation
    if nid in ("combined_defense", "combined"):
        return combined_defense.apply_combined_defense
    return _noop


def get_policy_prompt(name: Optional[str]) -> str | None:
    """Return the hardened system prompt for any prompt-level defense."""
    nid = (name or "").strip().lower()
    if nid in ("security_policy", "policy", "camel"):
        return security_policy.get_policy_prompt()
    if nid in ("defensive_token", "defense_token", "defense_tokens"):
        return defensive_token.get_defense_prompt()
    if nid in ("prompt_separation", "struq", "structured_query"):
        return prompt_separation.get_separation_prompt()
    if nid in ("combined_defense", "combined"):
        return combined_defense.get_combined_prompt()
    return None
