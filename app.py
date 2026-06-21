"""MindBridge — a two-audience depression decision-support prototype.

One flow, two readers:
  - Clinician: synthetic EEG + hormone data -> a REAL trained EEG model gives a
    calibrated, uncertainty-aware signal, plus guideline evidence to weigh (RAG).
  - Family: the same workup, turned into plain language + what-matters + next steps
    + trusted support resources -- help that is normally locked behind jargon.

Built for the USAII Global AI Hackathon 2026 (HS Challenge 1: "Help is Hard to Find").
Synthetic data only. Not a medical device. Decision SUPPORT -- the clinician decides.
"""

import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # read mindbridge/.env so an API key there is picked up automatically

from cases import DEMO_PATIENTS, patient_labels  # noqa: E402
from llm import hardcoded_analysis_response  # noqa: E402
from model_engine import analyze_patient  # noqa: E402
from rag import (  # noqa: E402
    guideline_retriever,
    resource_retriever,
)
from synthetic import archetype_label  # noqa: E402

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


@st.cache_resource
def retrievers():
    return guideline_retriever(), resource_retriever()


def disclaimer():
    st.markdown(
        """<div style="background:#f1f5f9;border-left:4px solid #2563eb;padding:10px 14px;
        border-radius:6px;font-size:0.9rem;line-height:1.45;">
        <b>Research prototype — not a medical device, not for clinical use.</b> All
        patients here are <b>synthetic</b>. The AI gives <b>decision support</b>, never a
        diagnosis or prescription — a clinician decides. The EEG model was trained on
        <b>adults</b>; adolescents are off-distribution.</div>""",
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
        {'· inside low-confidence band' if lo <= p <= hi else ''}</div>
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
        with st.expander(f"{len(result['ood_flags'])} feature(s) outside training range (OOD)"):
            for f in result["ood_flags"]:
                st.markdown(f"- {f}")


def clinician_view(out):
    c = out["clinician"]
    st.markdown(f"**Signal:** {c['signal_summary']}")
    st.markdown("**Evidence to consider** (from clinical guidelines):")
    for e in c["evidence_to_consider"]:
        st.markdown(f"- {e['point']}  \n  <span style='color:#64748b;font-size:0.85rem;'>"
                    f"Source: {e['source']}</span>", unsafe_allow_html=True)
    if c.get("safety_flags"):
        st.markdown("**Safety checks:**")
        for s in c["safety_flags"]:
            st.markdown(f"- {s}")
    st.info(c["caveat"])


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
    shown, resources = set(), []
    for r in res_retriever.always_show():
        _res_card(r); shown.add(r["id"]); resources.append(r)
    for r in res_retriever.by_ids(f.get("resource_ids", [])):
        if r["id"] not in shown:
            _res_card(r); shown.add(r["id"]); resources.append(r)
    st.download_button(
        "Download a take-home summary",
        _build_takehome(f, resources),
        file_name="mindbridge_summary.txt",
        use_container_width=True,
        help="A plain-language summary + checklist + resources to keep or share.",
    )


def _build_takehome(f, resources):
    """Turn the family explainer into a plain-text sheet the family can keep/print —
    understanding becomes a concrete artifact they leave with."""
    out = ["MindBridge — Your take-home summary", "=" * 38, "",
           "WHAT THIS MEANS", f["summary"], "", "WHAT MATTERS MOST"]
    out += [f"- {w}" for w in f["what_matters"]]
    out += ["", "YOUR NEXT STEPS"]
    out += [f"[ ] {s['step']}  ({s['timeframe']})" for s in f["next_steps"]]
    out += ["", "WHERE TO GET HELP"]
    out += [f"- {r['name']}: {r['contact']}  {r['url']}" for r in resources]
    out += ["",
            "If you or someone else is in danger right now, call or text 988",
            "(Suicide & Crisis Lifeline), or call 911.", "",
            "This summary is for understanding only — not medical advice. Your clinician",
            "makes the medical decisions. Generated by MindBridge (a research prototype)",
            "from synthetic data."]
    return "\n".join(out)


def _res_card(r):
    st.markdown(
        f"""<div style="border:1px solid #e2e8f0;border-radius:8px;padding:10px 12px;margin-bottom:6px;">
        <b>{r['name']}</b> — {r['description']}<br>
        <span style="font-size:0.9rem;">{r['contact']} · <a href="{r['url']}" target="_blank">{r['url']}</a></span></div>""",
        unsafe_allow_html=True,
    )


def _select_patient():
    """Friendly patient picker. Returns (seed, archetype, default_note, is_demo)."""
    choice = st.selectbox("Choose a patient", patient_labels())
    info = DEMO_PATIENTS[choice]
    st.caption(info["vignette"])
    return info["seed"], info["archetype"], info["note"], True


def main():
    st.set_page_config(page_title="MindBridge", layout="centered")
    st.title("MindBridge")
    st.markdown(
        "**For the parent or teen holding a mental-health workup they can't read.** "
        "After a depression evaluation, families are handed brainwave charts and clinical "
        "jargon — and no idea what to do next. MindBridge turns that workup into **plain "
        "language and clear next steps for the family**, while giving the clinician "
        "guideline-backed evidence. One analysis, both audiences."
    )
    disclaimer()
    _, r_ret = retrievers()

    st.markdown("### 1. Choose a patient")
    seed, arch, default_note, is_demo = _select_patient()
    result = analyze_patient(int(seed), arch)

    st.markdown("### 2. Synthetic inputs")
    show_inputs(result["patient"])

    st.markdown("### 3. Physician note")
    note = st.text_area("Physician note", value=default_note, height=180,
                        key=f"note_{seed}_{is_demo}", label_visibility="collapsed")

    run_key = (int(seed), arch, note)
    if st.button("Run AI analysis", type="primary", use_container_width=True):
        with st.spinner("Running analysis..."):
            time.sleep(2)
            out = hardcoded_analysis_response()
            note_excerpts = []
        # Persist so the page survives later clicks (download, checkboxes, decision log).
        st.session_state.analysis = {
            "key": run_key, "result": result, "out": out, "note_excerpts": note_excerpts,
        }

    saved = st.session_state.get("analysis")
    if saved and saved["key"] != run_key:
        st.info("Inputs changed — click **Run AI analysis** to refresh the results.")
    elif saved:
        _render_results(saved, r_ret)


def _render_results(saved, r_ret):
    result, out, note_excerpts = saved["result"], saved["out"], saved["note_excerpts"]

    st.markdown("### 4. Model signal")
    signal_card(result)
    if out.get("provider") == "presentation-demo":
        st.caption("Presentation demo response shown. The EEG model signal is local.")
    if note_excerpts:
        with st.expander(f"Note excerpts the AI used ({len(note_excerpts)}, retrieved via RAG)"):
            for ex in note_excerpts:
                st.markdown(f"- {ex}")

    st.markdown("### 5. Two audiences, one analysis")
    tab_f, tab_c = st.tabs(["Family view (plain language)", "Clinician view"])
    with tab_f:
        family_view(out, r_ret)
    with tab_c:
        clinician_view(out)

    # --- Human-in-the-loop ---
    st.divider()
    st.markdown("### 6. The clinician decides (human-in-the-loop)")
    st.caption("MindBridge does not diagnose or prescribe. Record the human decision:")
    case_id = result["patient"]["case_id"]
    d1, d2, d3 = st.columns(3)
    if "log" not in st.session_state:
        st.session_state.log = []
    if d1.button("Reviewed — agree", use_container_width=True):
        st.session_state.log.append(f"{case_id}: clinician agreed with support")
    if d2.button("Override", use_container_width=True):
        st.session_state.log.append(f"{case_id}: clinician overrode AI support")
    if d3.button("Needs more review", use_container_width=True):
        st.session_state.log.append(f"{case_id}: flagged for further review")
    for entry in st.session_state.log[-5:]:
        st.markdown(f"- {entry}")


if __name__ == "__main__":
    main()
