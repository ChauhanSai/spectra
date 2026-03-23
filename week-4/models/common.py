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
    last_pos = -1
    found = None
    for class_name in CLASSES:
        pos = normalized.find(class_name)
        if pos != -1 and pos > last_pos:
            last_pos = pos
            found = class_name
    if found is not None:
        return found
    norm_no_underscore = normalized.replace("_", "")
    for class_name in CLASSES:
        cn_flat = class_name.replace("_", "")
        if cn_flat in norm_no_underscore:
            pos = norm_no_underscore.find(cn_flat)
            if pos > last_pos:
                last_pos = pos
                found = class_name
    return found if found is not None else "unknown"
