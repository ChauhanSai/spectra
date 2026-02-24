import re

CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace(" ", "_")
    return text


def extract_label(text: str) -> str:
    if not (text or isinstance(text, str)):
        return "unknown"
    normalized = _normalize(text)
    for class_name in CLASSES:
        if class_name in normalized:
            return class_name
    for class_name in CLASSES:
        if class_name.replace("_", "") in normalized.replace("_", ""):
            return class_name
    return "unknown"
