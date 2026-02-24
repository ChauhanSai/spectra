import time
import base64
from pathlib import Path
from groq import Groq
from openai import OpenAI

import config


def get_client(model_key=None):
    """Return the appropriate API client for the given model key."""
    model_key = model_key or config.DEFAULT_MODEL
    model_info = config.MODEL_REGISTRY[model_key]
    backend = model_info["backend"]

    if backend == "ollama":
        return OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",  # required by client but unused
        )
    elif backend == "groq":
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found! Add it to .env file:\n"
                "  GROQ_API_KEY=your-key-here"
            )
        return Groq(api_key=config.GROQ_API_KEY)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_api_delay(model_key):
    """Ollama runs locally so no delay needed; cloud APIs need throttling."""
    model_info = config.MODEL_REGISTRY[model_key]
    if model_info["backend"] == "ollama":
        return 0
    return config.API_DELAY_SECONDS


def get_model_string(model_key):
    """Return the actual model identifier string to send to the API."""
    return config.MODEL_REGISTRY[model_key]["model"]


def classify_image(client, image_path, model_key=None):
    model_key = model_key or config.DEFAULT_MODEL
    model_str = get_model_string(model_key)
    model_info = config.MODEL_REGISTRY[model_key]
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

    if model_info["backend"] == "groq":
        response = client.chat.completions.create(
            model=model_str,
            messages=messages,
            max_tokens=50,
        )
    else:
        # Ollama / OpenAI-compatible
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
