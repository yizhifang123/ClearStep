"""Layer B — load Layer A artifacts for the dashboard (gracefully if absent).

Paths are relative to the project root; run Streamlit from there:
    .venv/bin/streamlit run app/Home.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ART = Path("ml/artifacts")
FEAT = Path("data/features")


def load_metrics():
    p = ART / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_model():
    # SECURITY: model.pkl is a LOCAL, self-generated artifact produced by
    # `ml/train.py` (joblib.dump) in this same repo — not an untrusted/external
    # file. joblib is the standard scikit-learn persistence format. Never point
    # this at a downloaded/third-party .pkl.
    p = ART / "model.pkl"
    if not p.exists():
        return None
    import joblib
    return joblib.load(p)


def load_subjects():
    p = FEAT / "mumtaz_subjects.csv"
    if not p.exists():
        return None
    import pandas as pd
    return pd.read_csv(p)


def feature_contributions(plain_pipeline, feature_names, x_row):
    """Linear, clinician-readable explanation: coefficient x standardized value.

    Returns [(feature, signed_contribution), ...] sorted by |contribution|.
    Positive pushes toward MDD; negative toward HC.
    """
    x_row = np.asarray(x_row, float).reshape(1, -1)
    pre = plain_pipeline[:-1]                      # impute + scale
    clf = plain_pipeline.named_steps["clf"]
    xs = pre.transform(x_row)[0]
    contrib = clf.coef_[0] * xs
    order = np.argsort(np.abs(contrib))[::-1]
    return [(feature_names[i], float(contrib[i])) for i in order]
