"""Cross-dataset transfer check (SCAFFOLD — NOT yet executed). NOT for clinical use.

Applies the EC-trained Mumtaz model (ml/artifacts/model.pkl) to a *second* open
dataset to test generalization. We EXPECT performance to drop — that is the honest
point: a high in-dataset AUC does not imply transfer to a new site/population.

STATUS: not run. It needs OpenNeuro **ds003478** downloaded (~9.4 GB — see
data/download.md), and the dataset-specific loader below
(`discover` / `labels_from_participants`) must be VERIFIED against the real BIDS
files before trusting any number (column names, file extension, BDI cutoff). The
model-application path reuses the exact Layer A feature pipeline, so once the
loader is confirmed this runs end to end.

Usage (after download + loader verification):
    .venv/bin/python -m ml.transfer_eval --data-root data/raw/ds003478
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from ml.features import epoch_feature_matrix
from ml.preprocess import process_raw

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("transfer")


def labels_from_participants(root, label_col="group", bdi_col="BDI", bdi_cutoff=7):
    """participant_id -> {0,1}. VERIFY against ds003478's participants.tsv.

    ds003478 defines "depressed" via self-report BDI in a college sample (not a
    clinical diagnosis). Prefer an explicit group column if present; otherwise
    threshold BDI. Adjust the column names / cutoff once you inspect the real file.
    """
    df = pd.read_csv(Path(root) / "participants.tsv", sep="\t")
    out = {}
    for _, r in df.iterrows():
        pid = str(r["participant_id"])
        if label_col in df.columns and pd.notna(r.get(label_col)):
            out[pid] = 1 if str(r[label_col]).lower() in {"mdd", "depressed", "dep", "case", "1"} else 0
        elif bdi_col in df.columns and pd.notna(r.get(bdi_col)):
            out[pid] = int(float(r[bdi_col]) >= bdi_cutoff)
    return out


def discover(root):
    """One EEG file per subject. VERIFY the glob/extension against the real layout."""
    root = Path(root)
    recs = []
    for ext in ("*_eeg.vhdr", "*_eeg.set", "*_eeg.edf"):
        for p in sorted(root.glob(f"sub-*/eeg/{ext}")):
            recs.append({"path": str(p), "subject": p.name.split("_")[0]})
    return recs


def _read(path: str):
    import mne
    if path.endswith(".vhdr"):
        return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")
    if path.endswith(".set"):
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")


def main():
    import joblib  # model.pkl is our own artifact (see app/model_io security note)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw/ds003478")
    ap.add_argument("--model", default="ml/artifacts/model.pkl")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model, names = bundle["calibrated"], bundle["feature_names"]
    labels = labels_from_participants(args.data_root)
    recs = discover(args.data_root)
    if not recs:
        raise SystemExit(f"No EEG files under {args.data_root}. See data/download.md.")

    X, y = [], []
    for r in recs:
        if r["subject"] not in labels:
            continue
        try:
            Xe, ns = epoch_feature_matrix(process_raw(_read(r["path"])))
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", r["subject"], e)
            continue
        idx = [ns.index(n) for n in names]            # align to the model's feature order
        X.append(np.nanmean(Xe[:, idx], axis=0))
        y.append(labels[r["subject"]])

    X, y = np.array(X), np.array(y)
    proba = model.predict_proba(X)[:, 1]
    out = {
        "dataset": str(args.data_root), "n": int(len(y)),
        "transfer_auc": float(roc_auc_score(y, proba)),
        "transfer_accuracy": float(accuracy_score(y, (proba >= 0.5).astype(int))),
    }
    Path("ml/artifacts/transfer.json").write_text(json.dumps(out, indent=2))
    log.info("Transfer: n=%d AUC=%.3f acc=%.3f",
             out["n"], out["transfer_auc"], out["transfer_accuracy"])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
