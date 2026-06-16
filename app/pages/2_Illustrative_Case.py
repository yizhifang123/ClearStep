"""Layer B page — ILLUSTRATIVE synthetic case. Simulated; NOT a validated prediction."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

import model_io
import synthetic
from ui import banner, hitl_controls, illustrative_badge, uncertainty_panel

st.set_page_config(page_title="Illustrative case (Layer B)", page_icon="🧪", layout="wide")
banner()
st.title("Illustrative case — envisioned workflow")
illustrative_badge()
st.error("**Illustrative only.** This page uses a SYNTHETIC patient. The hormonal inputs, "
         "subtype leaning, and treatment framing are simulated to show how the workflow "
         "*would* look — they are not validated predictions and cannot subtype a real "
         "person.")

case_choice = st.selectbox("Demo case", ["(custom)", *synthetic.CURATED_CASES], index=1)
if case_choice != "(custom)":
    seed, arch = synthetic.CURATED_CASES[case_choice]
    st.caption("Curated demo case. Pick '(custom)' to set the seed / archetype yourself.")
else:
    c1, c2 = st.columns(2)
    with c1:
        seed = st.number_input("Synthetic case seed", min_value=0, max_value=999,
                               value=7, step=1)
    with c2:
        arch = st.selectbox("Archetype", ["(random)", *synthetic.ARCHETYPES])
    arch = None if arch == "(random)" else arch
patient = synthetic.generate_synthetic_patient(int(seed), arch)

st.subheader(f"Patient {patient.case_id} — adolescent {patient.sex}, age {patient.age}")
st.caption(f"Hidden archetype (demo only): {synthetic.archetype_label(patient.archetype)}")

colA, colB = st.columns(2)
with colA:
    st.markdown("**EEG feature summary** (synthetic)")
    st.dataframe(pd.DataFrame({"feature": list(patient.eeg.keys()),
                               "value": [round(v, 3) for v in patient.eeg.values()]}),
                 height=300)
with colB:
    st.markdown("**Hormonal markers — synthetic cortisol** (calibrated to literature)")
    cort = patient.cortisol
    st.dataframe(pd.DataFrame({
        "marker": ["Morning (nmol/L)", "CAR rise (%)", "Evening (nmol/L)", "Diurnal slope"],
        "value": [round(cort["morning_nmol_l"], 1), round(cort["car_increase_pct"]),
                  round(cort["evening_nmol_l"], 1), round(cort["diurnal_slope"], 2)]}))
    st.caption("Reference (illustrative): morning ~12±6, CAR ~50%±25, evening ~3±1.5 nmol/L.")

st.divider()
st.subheader("Candidate profile — ILLUSTRATIVE")
lean = synthetic.illustrative_subtype_leaning(patient.cortisol)
b1, b2 = st.columns(2)
b1.metric("Melancholic leaning", f"{lean['melancholic']:.0%}")
b2.metric("Atypical leaning", f"{lean['atypical']:.0%}")
st.progress(lean["melancholic"])
st.warning("Subtype leaning here comes from a single illustrative cortisol mapping. In "
           "reality, cortisol cannot subtype an individual — huge overlap, and the "
           "dexamethasone suppression test failed for exactly this reason (RESEARCH.md §1.3).")

model = model_io.load_model()
if model is not None:
    x = patient.eeg_vector(model["feature_names"])
    p_mdd = float(model["calibrated"].predict_proba([x])[0, 1])
    st.subheader("MDD-vs-HC model applied to the synthetic EEG")
    uncertainty_panel(p_mdd, label="P(MDD) — Layer A model on SYNTHETIC input")
    st.caption("Illustrative: the model was trained on adults; applying it to a synthetic "
               "adolescent case demonstrates the workflow only.")
else:
    st.info("Train Layer A (`ml/train.py`) to also show the real MDD-vs-HC model applied "
            "to this synthetic EEG.")

st.divider()
st.subheader("Treatment framing — what this tool does NOT do")
st.markdown(
    "- It does **not** select or dose a medication.\n"
    "- The best-supported EEG treatment-response signal (rACC theta) is **prognostic, not "
    "prescriptive** — it predicts general improvement (including on placebo), not which drug.\n"
    "- Pharmacogenomic 'drug pickers' are contested and **showed no benefit in an "
    "adolescent RCT** (RESEARCH.md §1.4).")
st.divider()
hitl_controls(patient.case_id)
