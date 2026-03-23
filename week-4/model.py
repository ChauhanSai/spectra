import os
from models import get_predictor

predict_label = get_predictor(os.environ.get("SPECTRA_MODEL", "gemini"))
