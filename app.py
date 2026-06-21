"""MindBridge — a two-audience depression decision-support prototype.

One flow, two readers:
  • Clinician: synthetic EEG + hormone data -> a REAL trained EEG model gives a
    calibrated, uncertainty-aware signal, plus guideline evidence to weigh (RAG).
  • Family: the same workup, turned into plain language + what-matters + next steps
    + trusted support resources — help that is normally locked behind jargon.

Built for the USAII Global AI Hackathon 2026 (HS Challenge 1: "Help is Hard to Find").
Synthetic data only. Not a medical device. Decision SUPPORT — the clinician decides.
"""

import streamlit as st

from llm import NoKeyError, explain
from model_engine import DEMO_CASES, analyze_patient
from rag import (
    guideline_retriever,
    guidelines_block,
    resource_retriever,
    resources_block,
)
from synthetic import ARCHETYPES, archetype_label

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


@st.cache_resource
def retrievers():
    return guideline_retriever(), resource_retriever()


def disclaimer():
    st.warning(
        "⚠️ **Research prototype — not a medical device, not for clinical use.** All "
        "patients here are **synthetic**. The AI gives **decision support**, never a "
        "diagnosis or prescription — a clinician decides. The EEG model was trained on "
        "**adults**; adolescents are off-distribution.",
        icon="⚠️",
    )


def crisis_banner():
    st.markdown(
        """<div style="background:#991b1b;color:white;padding:10px 14px;border-radius:8px;">
        <b>In crisis right now?</b> Call or text <b>988</b> (Suicide &amp; Crisis Lifeline)
        or text <b>HOME to 741741</b>. If someone is unsafe, call 911.</div>""",
        unsafe_allow_html=True,
    )


def show_inputs(patient):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Brainwave summary (relative power, frontal region)**")
        frontal = {b: patient["eeg"][f"rel_{b}_frontal"] for b in BANDS}
        st.bar_chart(frontal, height=200)
        st.caption(
            f"Frontal alpha asymmetry (F4–F3): {patient['eeg']['faa_F4_F3']:+.2f} "
            "· negative = depression-leaning direction (exploratory)"
        )
    with c2:
        st.markdown("**Hormone panel (cortisol)**")
        cort = patient["cortisol"]
        st.metric("Morning (nmol/L)", f"{cort['morning_nmol_l']:.1f}")
        st.metric("Awakening response", f"{cort['car_increase_pct']:.0f}%")
        st.caption(f"Synthetic case {patient['case_id']} · age {patient['age']} · "
                   f"{patient['sex']} · {archetype_label(patient['archetype'])}")


def signal_card(result):
    p = result["p_mdd"]
    lo, hi = result["low_conf_band"]
    color = "#92400e" if lo <= p <= hi else ("#991b1b" if p > hi else "#166534")
    st.markdown(
        f"""<div style="border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px;">
        <div style="font-size:0.85rem;color:#64748b;">EEG model signal — pattern associated
        with MDD vs. healthy controls (real trained model)</div>
        <div style="font-size:2rem;font-weight:700;color:{color};">P = {p:.2f}</div>
        <div>Confidence: <b>{result['confidence']}</b>
        {'· ⚠️ inside low-confidence band' if lo <= p <= hi else ''}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    m = result["metrics"]
    ci = m["metric_ci_95"]["auc_loso"]
    st.caption(
        f"Model validation: LOSO AUC {m['auc_loso']:.2f} (95% CI {ci[0]:.2f}–{ci[1]:.2f}), "
        f"accuracy {m['accuracy_loso']:.0%}, permutation p={m['permutation_pvalue']:.3f}, "
        f"n={m['n_subjects']} adults. High accuracy on one small dataset ≠ clinical utility."
    )
    if result["ood_flags"]:
        with st.expander(f"⚠️ {len(result['ood_flags'])} feature(s) outside training range (OOD)"):
            for f in result["ood_flags"]:
                st.markdown(f"- {f}")


def clinician_view(out, guides):
    c = out["clinician"]
    st.markdown(f"**Signal:** {c['signal_summary']}")
    st.markdown("**Evidence to consider** (from clinical guidelines):")
    for e in c["evidence_to_consider"]:
        st.markdown(f"- {e['point']}  \n  <span style='color:#64748b;font-size:0.85rem;'>"
                    f"Source: {e['source']}</span>", unsafe_allow_html=True)
    if c.get("safety_flags"):
        st.markdown("**Safety checks:**")
        for s in c["safety_flags"]:
            st.markdown(f"- 🛡️ {s}")
    st.info(f"🔒 {c['caveat']}")


def family_view(out, res_retriever):
    f = out["family"]
    st.markdown(f"**What this means:** {f['summary']}")
    st.markdown("**What matters most:**")
    for w in f["what_matters"]:
        st.markdown(f"- {w}")
    st.markdown("**Your next steps:**")
    for i, s in enumerate(f["next_steps"], 1):
        st.checkbox(f"**{s['step']}**  ·  _{s['timeframe']}_", key=f"fstep_{i}")
    st.markdown("**Where to get help:**")
    shown = set()
    for r in res_retriever.always_show():
        _res_card(r); shown.add(r["id"])
    for r in res_retriever.by_ids(f.get("resource_ids", [])):
        if r["id"] not in shown:
            _res_card(r); shown.add(r["id"])


def _res_card(r):
    st.markdown(
        f"""<div style="border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;margin-bottom:6px;">
        <b>{r['name']}</b> — {r['description']}<br>
        <span style="font-size:0.9rem;">{r['contact']} · <a href="{r['url']}" target="_blank">{r['url']}</a></span></div>""",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="MindBridge", page_icon="🧠", layout="centered")
    st.title("🧠 MindBridge")
    st.markdown(
        "**The gap we close:** a clinician runs a mental-health workup, but the family "
        "is left with jargon they can't act on. MindBridge turns one depression "
        "decision-support analysis into **both** an evidence summary for the clinician "
        "**and** a plain-language explainer for the patient and family."
    )
    disclaimer()
    g_ret, r_ret = retrievers()

    st.markdown("### 1. Clinician: choose a synthetic patient")
    pick = st.selectbox("Demo patient", list(DEMO_CASES.keys()))
    seed, arch = DEMO_CASES[pick]
    with st.expander("…or build a custom synthetic patient"):
        seed = st.number_input("Seed", value=int(seed), step=1)
        arch = st.selectbox("Archetype", ARCHETYPES, index=ARCHETYPES.index(arch))
        st.caption("Custom patients need an ANTHROPIC_API_KEY for the written output.")

    result = analyze_patient(int(seed), arch)
    st.markdown("### 2. Synthetic inputs")
    show_inputs(result["patient"])

    if st.button("🤖 Run AI analysis", type="primary", use_container_width=True):
        with st.spinner("Running the EEG model and grounding in guidelines…"):
            st.markdown("### 3. Model signal")
            signal_card(result)

            q = (f"adolescent depression treatment safety {result['confidence']} signal "
                 f"{'high' if result['p_mdd'] > 0.5 else 'low'}")
            guides = g_ret.retrieve(q, top_k=5)
            resources = r_ret.retrieve(q, top_k=5)
            sig_txt = (f"P(MDD)={result['p_mdd']:.2f}, confidence={result['confidence']}, "
                       f"OOD_flags={len(result['ood_flags'])}, "
                       f"subtype_leaning(illustrative)={result['subtype_leaning']}")
            pat = result["patient"]
            pat_txt = f"age {pat['age']} {pat['sex']}, synthetic case {pat['case_id']}"

            is_demo = pick in DEMO_CASES and (int(seed), arch) == DEMO_CASES[pick]
            try:
                out = explain(pat_txt, sig_txt, guidelines_block(guides),
                              resources_block(resources),
                              demo_seed=int(seed) if is_demo else None)
            except NoKeyError as e:
                st.warning(str(e)); return
            except Exception as e:
                st.error(f"Analysis error: {e}"); return

        if out.get("mode") == "demo":
            st.caption("Demo mode: written output is cached (no API key). The model signal above is live.")

        st.markdown("### 4. Two audiences, one analysis")
        crisis_banner()
        tab_c, tab_f = st.tabs(["👩‍⚕️ Clinician view", "👨‍👩‍👧 Family view (plain language)"])
        with tab_c:
            clinician_view(out, guides)
        with tab_f:
            family_view(out, r_ret)

        # --- Human-in-the-loop ---
        st.divider()
        st.markdown("### 5. The clinician decides (human-in-the-loop)")
        st.caption("MindBridge does not diagnose or prescribe. Record the human decision:")
        d1, d2, d3 = st.columns(3)
        if "log" not in st.session_state:
            st.session_state.log = []
        if d1.button("✅ Reviewed — agree", use_container_width=True):
            st.session_state.log.append(f"{pat['case_id']}: clinician agreed with support")
        if d2.button("✏️ Override", use_container_width=True):
            st.session_state.log.append(f"{pat['case_id']}: clinician overrode AI support")
        if d3.button("🔎 Needs more review", use_container_width=True):
            st.session_state.log.append(f"{pat['case_id']}: flagged for further review")
        for entry in st.session_state.log[-5:]:
            st.markdown(f"- {entry}")


if __name__ == "__main__":
    main()
