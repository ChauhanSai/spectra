import time
import base64
from pathlib import Path
from groq import Groq
from openai import OpenAI

import config

# ── Lazy-loaded defended model (heavy, only load when needed) ──
_defended_model = None


def get_client(model_key=None):
    """Return the appropriate API client for the given model key."""
    model_key = model_key or config.DEFAULT_MODEL
    model_info = config.MODEL_REGISTRY[model_key]
    backend = model_info["backend"]

    if backend == "ollama":
        return OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",
        )
    elif backend == "groq":
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found! Add it to .env file:\n"
                "  GROQ_API_KEY=your-key-here"
            )
        return Groq(api_key=config.GROQ_API_KEY)
    elif backend == "transformers":
        return _get_defended_model(model_key)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _get_defended_model(model_key):
    """Lazy-load the DefensiveTokenModel (only once)."""
    global _defended_model
    if _defended_model is not None:
        return _defended_model

    from defend import DefensiveTokenModel

    model_info = config.MODEL_REGISTRY[model_key]
    hf_model = model_info.get("hf_model", config.DEFENSE_MODEL)

    _defended_model = DefensiveTokenModel(
        model_name=hf_model,
        num_tokens=config.DEFENSE_NUM_TOKENS,
    )

    # Load trained embeddings if available
    embed_path = config.DEFENSE_DIR / "optimized_embeddings.pt"
    if embed_path.exists():
        _defended_model.load_embeddings(embed_path)
    else:
        print("  ⚠ WARNING: No trained defensive embeddings found!")
        print(f"    Run train_defense.py first, or place embeddings at {embed_path}")

    return _defended_model


def get_api_delay(model_key):
    """Ollama and Transformers run locally — no delay needed."""
    model_info = config.MODEL_REGISTRY[model_key]
    if model_info["backend"] in ("ollama", "transformers"):
        return 0
    return config.API_DELAY_SECONDS


def get_model_string(model_key):
    """Return the actual model identifier string to send to the API."""
    return config.MODEL_REGISTRY[model_key]["model"]


def classify_image(client, image_path, model_key=None):
    model_key = model_key or config.DEFAULT_MODEL
    model_info = config.MODEL_REGISTRY[model_key]

    # ── Transformers backend (defensive tokens) ────────────────
    if model_info["backend"] == "transformers":
        return client.classify(image_path, use_defense=True)

    # ── Ollama / Groq backend ──────────────────────────────────
    model_str = get_model_string(model_key)
    image_path = Path(image_path)

    img_bytes = image_path.read_bytes()
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    suffix = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime_type = mime_map.get(suffix, "image/jpeg")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}",
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "What is the object in this image? "
                        "Reply with ONLY a single word or short phrase. "
                        "Do not explain."
                    ),
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model_str,
        messages=messages,
        max_tokens=50,
    )

    raw = response.choices[0].message.content.strip() if response.choices else ""
    return raw


def classify_with_retry(client, image_path, model_key=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            return classify_image(client, image_path, model_key)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait_time = min(15 * (2 ** attempt), 120)
                print(f"\n    ⏳ Rate limited. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...", end=" ")
                time.sleep(wait_time)
                print("Retrying...", end=" ")
            else:
                raise
    return classify_image(client, image_path, model_key)


def classify_batch(client, image_paths, model_key=None, delay=None, label=""):
    model_key = model_key or config.DEFAULT_MODEL
    delay = delay if delay is not None else get_api_delay(model_key)
    results = []
    total = len(image_paths)

    for i, path in enumerate(image_paths, 1):
        prefix = f"  [{label}]" if label else " "
        print(f"{prefix} [{i}/{total}] Classifying {Path(path).name}...", end=" ")

        try:
            response = classify_with_retry(client, path, model_key)
            print(f"→ {response!r}")
        except Exception as e:
            response = f"[ERROR] {e}"
            print(f"→ ERROR: {e}")

        results.append({
            "image_path": str(path),
            "response": response,
        })

        if i < total:
            time.sleep(delay)

    return results
