from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import model_io


def main() -> None:
    model = model_io.load_model()
    subjects = model_io.load_subjects()
    metrics = model_io.load_metrics()
    if model is None or subjects is None or metrics is None:
        raise SystemExit("Required Layer A artifacts are missing")

    feature_names = model["feature_names"]
    rows = []
    for _, row in subjects.iterrows():
        x_row = [float(row[name]) for name in feature_names]
        contributions = model_io.feature_contributions(model["plain"], feature_names, x_row)
        rows.append(
            {
                "id": str(row["subject"]),
                "label": "MDD" if int(row["label"]) == 1 else "HC",
                "probabilityMDD": round(float(row["oof_proba_mdd"]), 8),
                "contributions": [
                    {
                        "feature": name,
                        "value": round(float(value), 8),
                        "direction": "MDD" if value >= 0 else "HC",
                    }
                    for name, value in contributions[:8]
                ],
            }
        )

    honest = metrics["honest_loso"]
    bundle = {
        "schemaVersion": 1,
        "generatedAtUTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "condition": metrics["condition"],
        "metrics": {
            "nSubjects": int(honest["n_subjects"]),
            "nMdd": int(honest["n_mdd"]),
            "aucLoso": float(honest["auc_loso"]),
            "accuracyLoso": float(honest["accuracy_loso"]),
            "sensitivity": float(honest["sensitivity"]),
            "specificity": float(honest["specificity"]),
        },
        "subjects": rows,
    }

    out = ROOT / "native_macos" / "Resources" / "real_case_bundle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
