from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from week6_IGdefense.combo_runner import run_defense_combo_experiment


if __name__ == "__main__":
    run_defense_combo_experiment(
        model_id="llama3.2-vision-hf",
        results_csv=ROOT / "defensecombobymodel" / "llama" / "results_llama32_defense_combo.csv",
        max_images=50,
        api_delay_seconds=0.5,
    )
