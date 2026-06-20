"""Layer A — end-to-end pipeline runner. NOT for clinical use.

Usage (after deps installed and data in place — see data/download.md):
    .venv/bin/python -m ml.train --data-root data/raw/mumtaz --condition EC

Pipeline: discover EDFs -> preprocess -> per-epoch features -> save feature
tables -> honest leave-one-subject-out evaluation + the k-fold-vs-LOSO leakage
demo -> fit a final calibrated model for the app -> write metrics + figures to
ml/artifacts/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata as importlib_metadata
import json
import logging
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd

from ml.features import aggregate_by_subject, epoch_feature_matrix
from ml.preprocess import LABELS, discover_mumtaz, load_and_preprocess
from ml.validate import (
    calibration_cv_for_labels,
    honest_subject_evaluation,
    leakage_demo,
    make_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train")

ART = Path("ml/artifacts")
FIG = ART / "figures"
FEAT = Path("data/features")
SCHEMA_VERSION = 2
PREPROCESSING_PARAMS = {
    "l_freq": 1.0,
    "h_freq": 40.0,
    "notch": [50.0, 100.0],
    "epoch_len": 2.0,
    "overlap": 1.0,
    "reject_uv": 150.0,
    "do_ica": False,
}


def _package_versions() -> dict[str, str]:
    packages = ["mne", "numpy", "pandas", "scikit-learn", "scipy", "streamlit"]
    out = {}
    for package in packages:
        try:
            out[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            out[package] = "not-installed"
    return out


def _git_commit() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    commit = res.stdout.strip()
    return commit or None


def artifact_metadata(condition: str, data_root: str, feature_names: list[str],
                      preprocessing: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training": {
            "dataset": "Mumtaz 2016",
            "condition": condition,
            "data_root": data_root,
            "n_features": len(feature_names),
        },
        "preprocessing": preprocessing,
        "runtime": {
            "python": platform.python_version(),
            "packages": _package_versions(),
            "git_commit": _git_commit(),
        },
    }


def build_feature_reference(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """Training distribution summary used for simple OOD/data-quality flags."""
    reference = {}
    for name in feature_names:
        values = pd.to_numeric(df[name], errors="coerce").dropna().to_numpy(float)
        reference[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p01": float(np.percentile(values, 1)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return reference


def build_feature_table(data_root: str, condition: str, min_epochs: int = 5):
    records = [r for r in discover_mumtaz(data_root) if r["condition"] == condition]
    if not records:
        raise SystemExit(
            f"No EDF files for condition={condition} under {data_root}. "
            "See data/download.md."
        )
    rows, subjects, labels, names = [], [], [], None
    for r in records:
        try:
            epochs = load_and_preprocess(r["path"])
        except Exception as e:  # noqa: BLE001
            log.warning("Failed on %s: %s", r["path"], e)
            continue
        if len(epochs) < min_epochs:
            log.warning("Only %d epochs for %s; skipping.", len(epochs), r["subject"])
            continue
        X, names = epoch_feature_matrix(epochs)
        rows.append(X)
        subjects += [r["subject"]] * len(X)
        labels += [LABELS[r["label"]]] * len(X)
        log.info("%-10s (%s): %d epochs", r["subject"], r["label"], len(X))
    if not rows:
        raise SystemExit("No usable recordings after preprocessing.")
    df = pd.DataFrame(np.vstack(rows), columns=names)
    df.insert(0, "label", labels)
    df.insert(0, "subject", subjects)
    return df, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw/mumtaz")
    ap.add_argument("--condition", default="EC", choices=["EC", "EO", "TASK"])
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()

    FEAT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df, names = build_feature_table(args.data_root, args.condition)
    df.to_csv(FEAT / "mumtaz_epochs.csv", index=False)
    log.info("Saved epoch features: %d epochs x %d features", len(df), len(names))

    X_ep = df[names].to_numpy(float)
    y_ep = df["label"].to_numpy(int)
    groups = df["subject"].to_numpy()

    X_subj, order = aggregate_by_subject(X_ep, groups)
    lab_map = dict(zip(df["subject"], df["label"]))
    y_subj = np.array([lab_map[s] for s in order], int)
    log.info("Subjects: %d (MDD=%d, HC=%d)", len(y_subj), int(y_subj.sum()),
             int((y_subj == 0).sum()))

    honest = honest_subject_evaluation(X_subj, y_subj, n_permutations=args.permutations)
    leak = leakage_demo(X_ep, y_ep, groups)
    metadata = artifact_metadata(args.condition, args.data_root, names, PREPROCESSING_PARAMS)
    metrics = {"schema_version": SCHEMA_VERSION, "condition": args.condition,
               "features": names, "metadata": metadata,
               "honest_loso": honest, "leakage_demo": leak}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Honest LOSO: AUC=%.3f acc=%.3f (majority=%.3f, perm p=%.3f)",
             honest["auc_loso"], honest["accuracy_loso"],
             honest["majority_class_accuracy"], honest["permutation_pvalue"])
    log.info("Leakage demo: leaky acc=%.3f vs honest acc=%.3f (delta=%.3f)",
             leak["leaky_accuracy"], leak["grouped_accuracy"], leak["delta_accuracy"])

    # honest out-of-fold probabilities for the per-subject table (used by the app)
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    oof = cross_val_predict(make_pipeline(), X_subj, y_subj, cv=LeaveOneOut(),
                            method="predict_proba")[:, 1]
    subj_df = pd.DataFrame(X_subj, columns=names)
    subj_df.insert(0, "oof_proba_mdd", oof)
    subj_df.insert(0, "label", y_subj)
    subj_df.insert(0, "subject", order)
    subj_df.to_csv(FEAT / "mumtaz_subjects.csv", index=False)
    feature_reference = build_feature_reference(subj_df, names)

    # final calibrated model for the app + a plain pipeline for coefficient explanations
    from sklearn.calibration import CalibratedClassifierCV
    calibration_cv = calibration_cv_for_labels(y_subj, requested_splits=5, random_state=0)
    calibrated = CalibratedClassifierCV(
        make_pipeline(), method="sigmoid",
        cv=calibration_cv)
    calibrated.fit(X_subj, y_subj)
    plain = make_pipeline().fit(X_subj, y_subj)
    joblib.dump({
        "schema_version": SCHEMA_VERSION,
        "calibrated": calibrated,
        "plain": plain,
        "feature_names": names,
        "feature_reference": feature_reference,
        "labels": {"0": "HC", "1": "MDD"},
        "metrics": metrics,
        "metadata": metadata,
        "calibration": {
            "method": "sigmoid",
            "cv": (
                f"StratifiedKFold(n_splits={calibration_cv.n_splits}, "
                "shuffle=True, random_state=0)"
            ),
        },
    }, ART / "model.pkl")
    log.info("Saved model + metrics to %s", ART)

    _figures(honest, leak, y_subj, oof)
    log.info("Done. Figures in %s", FIG)


def _figures(honest, leak, y_subj, oof):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(["leaky\nk-fold", "honest\nsubject-wise"],
           [leak["leaky_accuracy"], leak["grouped_accuracy"]],
           color=["#c0392b", "#27ae60"])
    ax.axhline(honest["majority_class_accuracy"], ls="--", c="gray", label="majority class")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title(f"Leakage inflation: delta acc = {leak['delta_accuracy']:.2f}")
    fig.tight_layout(); fig.savefig(FIG / "leakage_demo.png", dpi=130); plt.close(fig)

    tn, fp, fn, tp = honest["confusion_tn_fp_fn_tp"]
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HC", "MDD"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["HC", "MDD"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"LOSO confusion (AUC={honest['auc_loso']:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    RocCurveDisplay.from_predictions(y_subj, oof, ax=ax, name="LOSO")
    ax.plot([0, 1], [0, 1], ls="--", c="gray")
    ax.set_title("ROC (leave-one-subject-out)")
    fig.tight_layout(); fig.savefig(FIG / "roc_curve.png", dpi=130); plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_subj, oof, n_bins=5, strategy="quantile")
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.plot(mean_pred, frac_pos, "o-"); ax.plot([0, 1], [0, 1], ls="--", c="gray")
    ax.set_xlabel("mean predicted prob"); ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (LOSO)")
    fig.tight_layout(); fig.savefig(FIG / "calibration_curve.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
