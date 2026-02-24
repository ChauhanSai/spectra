from __future__ import annotations

from typing import Callable

from PIL import Image

from . import clarifai_minicpm
from . import clarifai_ministral
from . import claude
from . import gemini
from . import hf_gemma
from . import hf_internvl
from . import hf_llama
from . import local_gemma
from . import local_gemma3n
from . import openai
from . import local_internvl
from . import local_llava
from . import local_llama32_vision
from . import local_minicpm
from . import moondream
from . import openrouter_gemma
from . import openrouter_internvl
from . import qwen
from . import replicate_internvl

Predictor = Callable[[Image.Image], str]

REGISTRY: dict[str, Predictor] = {
    "claude": claude.predict_label,
    "claude-sonnet-4": claude.predict_label,
    "gemini": gemini.predict_label,
    "gemini-2.5-flash": gemini.predict_label,
    "gemini-2.5-pro": gemini.predict_label,
    "gemini-3-flash-preview": gemini.predict_label,
    "gemini-3-pro": gemini.predict_label,
    "gpt-4o": openai.predict_label,
    "gpt-4o-mini": openai.predict_label,
    "openai": openai.predict_label,
    "qwen": qwen.predict_label,
    "qwen2.5-vl-3b-instruct": qwen.predict_label,
    "gemma-3": openrouter_gemma.predict_label,
    "gemma-3-12b-it:free": openrouter_gemma.predict_label,
    "gemma-3-local": local_gemma.predict_label,
    "gemma-3-4b-it-local": local_gemma.predict_label,
    "gemma-3n": local_gemma3n.predict_label,
    "gemma-3n-2b": local_gemma3n.predict_label,
    "gemma-3-hf": hf_gemma.predict_label,
    "llama4-scout": hf_llama.predict_label,
    "hf-llama4-scout": hf_llama.predict_label,
    "internvl3-38b": hf_internvl.predict_label,
    "internvl3": hf_internvl.predict_label,
    "internvl3-30b": replicate_internvl.predict_label,
    "replicate-internvl": replicate_internvl.predict_label,
    "internvl3-2b": openrouter_internvl.predict_label,
    "openrouter-internvl": openrouter_internvl.predict_label,
    "internvl3-2b-local": local_internvl.predict_label,
    "local-internvl": local_internvl.predict_label,
    "llava-1.6-13b": local_llava.predict_label,
    "local-llava": local_llava.predict_label,
    "llama3.2-vision": local_llama32_vision.predict_label,
    "llama-3.2-vision": local_llama32_vision.predict_label,
    "moondream": moondream.predict_label,
    "minicpm": clarifai_minicpm.predict_label,
    "clarifai-minicpm": clarifai_minicpm.predict_label,
    "minicpm-v2.6": local_minicpm.predict_label,
    "minicpm-v2.6-local": local_minicpm.predict_label,
    "local-minicpm": local_minicpm.predict_label,
    "ministral": clarifai_ministral.predict_label,
    "clarifai-ministral": clarifai_ministral.predict_label,
}


def list_models() -> list[str]:
    return sorted(REGISTRY.keys())


def get_predictor(model_id: str) -> Predictor:
    model_id = (model_id or "gemini").strip().lower()
    if model_id not in REGISTRY:
        raise ValueError(f"Unknown model: {model_id!r}. Available: {', '.join(list_models())}")
    return REGISTRY[model_id]
