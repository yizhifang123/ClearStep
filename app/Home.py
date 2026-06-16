"""Layer B — clinician decision-support dashboard (Streamlit entry point).
Research prototype. NOT for clinical use.

Run from the project root:
    .venv/bin/streamlit run app/Home.py
"""
import streamlit as st

import model_io
from ui import DISCLAIMER, banner

st.set_page_config(page_title="Depression Tx Decision-Support (Prototype)",
                   page_icon="🧠", layout="wide")

banner()
st.title("🧠 Personalized Depression Treatment — Decision-Support")
st.caption("A research prototype exploring how EEG + hormonal signals *could* support a "
           "psychiatrist's treatment decisions — with honest uncertainty and the "
           "clinician always in control.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Layer A — real ML slice")
    st.markdown(
        "- Trained on the **open Mumtaz 2016** EEG dataset\n"
        "- The task the data actually supports: **MDD vs. healthy control** "
        "(*not* treatment response — no open data supports that)\n"
        "- **Subject-wise** validation + a built-in **leakage demonstration**\n"
        "- See the **Real case (Layer A)** page →")
    m = model_io.load_metrics()
    if m:
        h, l = m["honest_loso"], m["leakage_demo"]
        a, b = st.columns(2)
        a.metric("Honest LOSO AUC", f"{h['auc_loso']:.2f}")
        b.metric("Leakage Δ accuracy",
                 f"{l['delta_accuracy']:+.0%}",
                 help=f"Leaky k-fold {l['leaky_accuracy']:.0%} vs honest "
                      f"subject-wise {l['grouped_accuracy']:.0%}.")
    else:
        st.info("Layer A artifacts not found yet. Place data (see `data/download.md`) "
                "then run `.venv/bin/python -m ml.train`.")

with col2:
    st.subheader("Layer B — illustrative workflow")
    st.markdown(
        "- **Synthetic** adolescent-male cases (the target population)\n"
        "- EEG summary + **synthetic cortisol** → an **illustrative** subtype / "
        "treatment *leaning*\n"
        "- Prominent uncertainty, plain-language explanations, **human-in-the-loop**\n"
        "- See the **Illustrative case (Layer B)** page →")
    st.warning("Everything in Layer B beyond the Layer A MDD-vs-HC model is "
               "**simulated and illustrative** — not a validated prediction.")

st.divider()
st.markdown("#### What's real vs. illustrative")
st.markdown(
    "| Element | Status |\n|---|---|\n"
    "| EEG → MDD-vs-HC probability (real Mumtaz subjects) | ✅ **Real** (Layer A) |\n"
    "| Subject-wise metrics + leakage demonstration | ✅ **Real** |\n"
    "| Hormonal / cortisol inputs | 🟠 **Synthetic** (calibrated to literature) |\n"
    "| Subtype leaning (melancholic / atypical) | 🟠 **Illustrative** |\n"
    "| Treatment-response framing | 🟠 **Illustrative** (prognostic, *not* a drug picker) |")
st.caption(DISCLAIMER)
