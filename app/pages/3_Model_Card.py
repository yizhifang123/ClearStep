"""Layer B page — Model Card / limitations. NOT for clinical use."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

import model_io
from ui import banner

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

h, l = metrics["honest_loso"], metrics["leakage_demo"]
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

st.subheader("The leakage demonstration")
st.markdown(
    f"Same model, two validation schemes:\n"
    f"- Leaky random k-fold: **{l['leaky_accuracy']:.0%}** accuracy\n"
    f"- Honest subject-wise: **{l['grouped_accuracy']:.0%}** accuracy\n"
    f"- Inflation attributable to leakage: **{l['delta_accuracy']:+.0%}**\n\n"
    "The well-known '~98% MDD detection' headlines arise from the leaky kind of split — "
    "reported on *this very dataset* (Mumtaz 2018). We report the honest number.")

FIG = Path("ml/artifacts/figures")
cols = st.columns(2)
figs = [("Leakage inflation", "leakage_demo.png"), ("Confusion (LOSO)", "confusion_matrix.png"),
        ("ROC (LOSO)", "roc_curve.png"), ("Calibration (LOSO)", "calibration_curve.png")]
for i, (title, fn) in enumerate(figs):
    p = FIG / fn
    if p.exists():
        cols[i % 2].image(str(p), caption=title)

st.subheader("Key limitations (honest)")
st.markdown(
    "- **Frontal alpha asymmetry is not a validated diagnostic biomarker** (meta-analytic "
    "d ≈ −0.007); it is included as one *exploratory* feature, never a standalone driver.\n"
    "- **Adults only** — applying the model to the adolescent target population is "
    "off-distribution.\n"
    "- **Small sample** (tens of subjects) → wide confidence intervals; treat metrics as "
    "indicative, not definitive.\n"
    "- **Diagnosis, not treatment response** — no open dataset supports treatment-response "
    "prediction.\n"
    "- **Regulatory:** an EEG-analyzing tool processes a physiological signal, so for real "
    "clinical use it would be a **medical device**, *not* exempt 'Non-Device CDS'. This "
    "prototype is research-only.\n"
    "- Hormonal, subtype, and treatment panels are **synthetic and illustrative**.")
