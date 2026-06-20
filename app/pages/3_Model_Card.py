"""Layer B page — Model Card / limitations. NOT for clinical use."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

import model_io
from ui import banner


def _ci_text(metrics: dict, key: str) -> str | None:
    ci = metrics.get("metric_ci_95", {}).get(key)
    if not ci:
        return None
    return f"{ci[0]:.2f}-{ci[1]:.2f}"


st.set_page_config(page_title="Model card", page_icon="📋", layout="wide")
banner()
st.title("📋 Model card & limitations")

st.subheader("Intended use")
st.markdown(
    "- **Intended:** a research / education demonstration of an honest EEG → MDD-vs-HC "
    "pipeline and a decision-support *workflow*.\n"
    "- **Not intended:** diagnosis, treatment selection, dosing, or any clinical use; "
    "use with adolescents (training data is adults); use as a self-assessment tool.")

metrics = model_io.load_metrics()
if not metrics:
    st.info("Run `.venv/bin/python -m ml.train` to populate honest metrics and figures.")
    st.stop()

h, leak = metrics["honest_loso"], metrics["leakage_demo"]
st.subheader("Layer A — honest performance (leave-one-subject-out)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC-AUC", f"{h['auc_loso']:.2f}")
m2.metric("Accuracy", f"{h['accuracy_loso']:.0%}",
          help=f"Majority-class floor {h['majority_class_accuracy']:.0%}")
m3.metric("Sensitivity", f"{h['sensitivity']:.0%}")
m4.metric("Specificity", f"{h['specificity']:.0%}")
st.caption(f"n={h['n_subjects']} subjects ({h['n_mdd']} MDD). Permutation test "
           f"p={h['permutation_pvalue']:.3f} (chance AUC = 0.50). "
           f"Resting condition: {metrics['condition']}.")
ci_bits = [
    f"AUC {_ci_text(h, 'auc_loso')}",
    f"accuracy {_ci_text(h, 'accuracy_loso')}",
    f"sensitivity {_ci_text(h, 'sensitivity')}",
    f"specificity {_ci_text(h, 'specificity')}",
]
if h.get("metric_ci_95"):
    st.caption("Subject-bootstrap 95% intervals: " + "; ".join(ci_bits) + ".")
if h.get("permutation_cv"):
    st.caption(f"Permutation CV: {h['permutation_cv']}. {h.get('permutation_note', '')}")
st.warning(
    "**High accuracy ≠ clinical utility.** This is validated without known subject-level "
    "leakage in this dataset (subject-wise + "
    "permutation p < 0.01), but Mumtaz is a small, single-site, adult dataset known to "
    "over-separate. The classifier is driven mainly by **relative beta power** (see "
    "drivers below) — which can reflect cortical hyperarousal *or* a non-neural confound "
    "(EMG / muscle tension, acquisition differences); we cannot disambiguate from this "
    "data. It is **not** validated for generalization, for adolescents, or for any "
    "clinical use.")

st.subheader("The leakage demonstration")
st.markdown(
    f"Same model + features, two epoch-level validation schemes:\n"
    f"- Leaky random k-fold (subjects bleed across folds): **{leak['leaky_accuracy']:.0%}** accuracy\n"
    f"- Honest subject-wise (GroupKFold): **{leak['grouped_accuracy']:.0%}** accuracy\n"
    f"- Inflation attributable to leakage: **{leak['delta_accuracy']:+.0%}**\n\n"
    "Leakage inflates accuracy in the expected direction. The gap is *modest* here "
    "because our 16 aggregated features have little capacity to memorize individual "
    "subjects — the dramatic ~98%→chance collapses in the literature come from "
    "high-dimensional / deep models on per-epoch data. The lesson stands: always "
    "validate subject-wise.")
st.caption(
    "Granularity note: the headline metrics above are **subject-level** (each subject's "
    "epochs averaged to one vector, then leave-one-subject-out — denoised, hence higher); "
    "this leakage demo runs at the noisier **epoch level** to make subject bleed-through "
    "concrete.")

FIG = Path("ml/artifacts/figures")
cols = st.columns(2)
figs = [("Leakage inflation", "leakage_demo.png"), ("Confusion (LOSO)", "confusion_matrix.png"),
        ("ROC (LOSO)", "roc_curve.png"), ("Calibration (LOSO)", "calibration_curve.png")]
for i, (title, fn) in enumerate(figs):
    p = FIG / fn
    if p.exists():
        cols[i % 2].image(str(p), caption=title)

model = model_io.load_model()
if model is not None:
    coef = model["plain"].named_steps["clf"].coef_[0]
    names = model["feature_names"]
    order = np.argsort(np.abs(coef))[::-1][:6]
    st.subheader("What drives the model")
    st.markdown("Top standardized coefficients (＋ → toward MDD):")
    st.dataframe(pd.DataFrame({"feature": [names[i] for i in order],
                               "coefficient": [round(float(coef[i]), 2) for i in order]}))
    st.caption("Relative **beta** power dominates; frontal alpha asymmetry contributes "
               "little — consistent with FAA being a near-null biomarker. Beta-driven "
               "separation is exactly why the confound caveat above matters.")
    if model.get("calibration"):
        st.caption(f"App model calibration: {model['calibration']['method']} via "
                   f"{model['calibration']['cv']}.")

rob_path = Path("ml/artifacts/robustness.json")
if rob_path.exists():
    import json
    rob = json.loads(rob_path.read_text())
    st.subheader("Robustness — eyes-closed vs. eyes-open")
    st.dataframe(pd.DataFrame([
        {"condition": c, "n": v["n_subjects"], "AUC": round(v["auc_loso"], 2),
         "accuracy": round(v["accuracy_loso"], 2),
         "sensitivity": round(v["sensitivity"], 2),
         "specificity": round(v["specificity"], 2)}
        for c, v in rob.items()]))
    st.caption("The MDD-vs-HC signal persists across both resting conditions (a good "
               "robustness sign) — though a *stable* confound (site/acquisition) would "
               "also be condition-robust, so this does not rule one out.")

transfer = model_io.load_transfer()
st.subheader("External transfer validation")
if transfer:
    t1, t2, t3 = st.columns(3)
    t1.metric("External n", transfer["n"])
    t2.metric("Transfer AUC", f"{transfer['transfer_auc']:.2f}")
    t3.metric("Transfer accuracy", f"{transfer['transfer_accuracy']:.0%}")
    st.caption(f"Dataset: {transfer.get('dataset', 'external cohort')}. "
               "This is the strongest generalization check and should be expected "
               "to drop versus in-dataset Mumtaz performance.")
else:
    st.info("Not run yet. Download OpenNeuro ds003478 and run "
            "`.venv/bin/python -m ml.transfer_eval --data-root data/raw/ds003478` "
            "after verifying its BIDS labels and EEG file layout.")

st.subheader("Key limitations (honest)")
st.markdown(
    "- **Frontal alpha asymmetry is not a validated diagnostic biomarker** (meta-analytic "
    "d ≈ −0.007); included as one *exploratory* feature — and indeed it barely contributes.\n"
    "- **Beta-dominated signal** may be a non-neural confound (EMG / acquisition), not "
    "depression — not disambiguated here.\n"
    "- **Adults only** — applying the model to the adolescent target population is "
    "off-distribution.\n"
    "- **Small, single-site sample** (n=58) → wide confidence intervals; no external "
    "validation, so high in-sample accuracy does **not** imply generalization.\n"
    "- **Diagnosis, not treatment response** — no open dataset supports treatment-response "
    "prediction.\n"
    "- **Regulatory:** an EEG-analyzing tool processes a physiological signal, so for real "
    "clinical use it would be a **medical device**, *not* exempt 'Non-Device CDS'. This "
    "prototype is research-only.\n"
    "- Hormonal, subtype, and treatment panels are **synthetic and illustrative**.")
