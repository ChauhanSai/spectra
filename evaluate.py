import time
import base64
from pathlib import Path
from groq import Groq

import config


def get_client():
    return Groq(api_key=config.GROQ_API_KEY)


def classify_image(client, image_path):
    image_path = Path(image_path)

    img_bytes = image_path.read_bytes()
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    suffix = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime_type = mime_map.get(suffix, "image/jpeg")

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
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
        ],
        max_tokens=50,
    )

    raw = response.choices[0].message.content.strip() if response.choices else ""
    return raw


def classify_with_retry(client, image_path, max_retries=5):
    for attempt in range(max_retries):
        try:
            return classify_image(client, image_path)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                wait_time = min(15 * (2 ** attempt), 120)
                print(f"\n    ⏳ Rate limited. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...", end=" ")
                time.sleep(wait_time)
                print("Retrying...", end=" ")
            else:
                raise
    return classify_image(client, image_path)


def classify_batch(client, image_paths, delay=None, label=""):
    delay = delay if delay is not None else config.API_DELAY_SECONDS
    results = []
    total = len(image_paths)

    for i, path in enumerate(image_paths, 1):
        prefix = f"  [{label}]" if label else " "
        print(f"{prefix} [{i}/{total}] Classifying {Path(path).name}...", end=" ")

        try:
            response = classify_with_retry(client, path)
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
