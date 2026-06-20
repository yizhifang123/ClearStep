"""Layer B page — REAL case (Layer A model on open EEG data). NOT for clinical use."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import model_io
from ui import banner, contributions_chart, hitl_controls, real_badge, uncertainty_panel

st.set_page_config(page_title="Real case (Layer A)", page_icon="🧠", layout="wide")
banner()
st.title("Real case — Layer A model")
real_badge()
st.caption("EEG → MDD-vs-healthy-control probability from the model trained on the open "
           "Mumtaz 2016 dataset. This is the genuinely real ML slice; metrics shown are "
           "leave-one-subject-out (honest). It classifies **diagnosis status, not "
           "treatment response**.")

model = model_io.load_model()
subjects = model_io.load_subjects()
metrics = model_io.load_metrics()
if model is None or subjects is None:
    st.info("Layer A artifacts not found yet. Place the Mumtaz data (see "
            "`data/download.md`) then run `.venv/bin/python -m ml.train`. "
            "This page populates automatically.")
    st.stop()

feature_names = model["feature_names"]
sid = st.selectbox("Select a held-out subject", subjects["subject"].tolist())
row = subjects[subjects["subject"] == sid].iloc[0]
proba = float(row["oof_proba_mdd"])
true_label = "MDD" if int(row["label"]) == 1 else "HC"

left, right = st.columns(2)
with left:
    st.subheader("Model suggestion")
    uncertainty_panel(proba, label="P(MDD) — honest LOSO estimate")
    st.caption(f"Ground-truth label (demo transparency only): **{true_label}**")
    if metrics:
        h = metrics["honest_loso"]
        st.caption(f"Model context: LOSO AUC {h['auc_loso']:.2f}, "
                   f"sensitivity {h['sensitivity']:.0%}, specificity {h['specificity']:.0%}. "
                   "Full detail on the Model Card page.")
with right:
    st.subheader("Why — feature contributions")
    x_row = [float(row[f]) for f in feature_names]
    flags = model_io.feature_quality_flags(model, feature_names, x_row)
    if flags:
        st.warning("Data-quality / OOD flags: " + "; ".join(flags[:5]))
    contribs = model_io.feature_contributions(model["plain"], feature_names, x_row)
    contributions_chart(contribs)

st.divider()
hitl_controls(sid)
