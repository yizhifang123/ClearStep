"""Robustness check — honest LOSO performance across resting conditions.

Reuses the Layer A pipeline to evaluate eyes-closed (EC) vs eyes-open (EO) and
writes ml/artifacts/robustness.json — WITHOUT touching the production artifacts
that train.py wrote. A consistent EC/EO result is a small honesty signal; a big
divergence would flag condition-specific (possibly artefactual) effects.

Usage: .venv/bin/python -m ml.robustness
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ml.features import aggregate_by_subject, epoch_feature_matrix
from ml.preprocess import LABELS, discover_mumtaz, load_and_preprocess
from ml.validate import honest_subject_evaluation, leakage_demo

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("robustness")
ART = Path("ml/artifacts")


def features_for_condition(data_root, condition, min_epochs=5):
    recs = [r for r in discover_mumtaz(data_root) if r["condition"] == condition]
    if not recs:
        raise SystemExit(
            f"No EDF files for condition={condition} under {data_root}. See data/download.md."
        )
    rows, subjects, labels = [], [], []
    for r in recs:
        try:
            ep = load_and_preprocess(r["path"])
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", r["path"], e)
            continue
        if len(ep) < min_epochs:
            continue
        X, _ = epoch_feature_matrix(ep)
        rows.append(X)
        subjects += [r["subject"]] * len(X)
        labels += [LABELS[r["label"]]] * len(X)
    if not rows:
        raise SystemExit(
            f"No usable recordings for condition={condition} under {data_root} after preprocessing."
        )
    Xep = np.vstack(rows)
    groups = np.array(subjects)
    Xs, order = aggregate_by_subject(Xep, groups)
    lab = dict(zip(subjects, labels))
    ys = np.array([lab[s] for s in order])
    return Xep, np.array(labels), groups, Xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw/mumtaz")
    ap.add_argument("--permutations", type=int, default=500)
    args = ap.parse_args()

    out = {}
    for cond in ("EC", "EO"):
        log.info("=== condition %s ===", cond)
        Xep, yep, groups, Xs, ys = features_for_condition(args.data_root, cond)
        honest = honest_subject_evaluation(Xs, ys, n_permutations=args.permutations)
        leak = leakage_demo(Xep, yep, groups)
        out[cond] = {
            "n_subjects": int(len(ys)), "n_mdd": int(ys.sum()),
            "auc_loso": honest["auc_loso"], "accuracy_loso": honest["accuracy_loso"],
            "sensitivity": honest["sensitivity"], "specificity": honest["specificity"],
            "permutation_pvalue": honest["permutation_pvalue"],
            "leak_delta_accuracy": leak["delta_accuracy"],
        }
        log.info("%s: AUC=%.3f acc=%.3f (p=%.3f)", cond,
                 honest["auc_loso"], honest["accuracy_loso"], honest["permutation_pvalue"])
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
