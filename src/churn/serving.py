"""Save, load and serve the champion model.

The model is stored with skops rather than pickle: loading a pickle runs whatever
code is inside it, while skops refuses any type that is not on an explicit list.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_FILE = "churn_model.skops"
META_FILE = "model_meta.json"

# The only non-standard objects the saved pipeline may contain: our own two
# row-by-row feature functions. Anything else makes loading fail.
TRUSTED_TYPES = ["churn.features.add_features", "churn.features.to_float32"]


def save_model(pipeline, models_dir: Path, results: dict) -> None:
    import skops.io as sio

    models_dir.mkdir(parents=True, exist_ok=True)
    sio.dump(pipeline, models_dir / MODEL_FILE)
    meta = {
        "version": results["version"],
        "model": results["champion"],
        "data_sha256": results["data"]["sha256"],
        "test_scores": results["champion_test"]["scores"],
        "offer_assumptions": results["offer_assumptions"],
        "versions": results["versions"],
    }
    (models_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")


def load_model(models_dir: Path):
    import skops.io as sio

    path = models_dir / MODEL_FILE
    unknown = set(sio.get_untrusted_types(file=path)) - set(TRUSTED_TYPES)
    if unknown:
        raise ValueError(f"Refusing to load {path}: unexpected types {sorted(unknown)}")
    pipeline = sio.load(path, trusted=TRUSTED_TYPES)
    meta = json.loads((models_dir / META_FILE).read_text())
    return pipeline, meta
