"""Layer B — shared Streamlit UI components. Research prototype; NOT for clinical use."""
from __future__ import annotations

import streamlit as st

DISCLAIMER = (
    "⚠️ **Research prototype — NOT for clinical use.** It does not diagnose, "
    "prescribe, or make validated medical claims. Decision *support* only — the "
    "clinician decides. Not validated in adolescents (training data is adults)."
)

# Predicted-probability band treated as "insufficient evidence".
LOW_CONF_BAND = (0.40, 0.60)


def banner():
    st.warning(DISCLAIMER)


def confidence_state(p: float):
    """Return (label, flag_for_review) for a probability."""
    if LOW_CONF_BAND[0] <= p <= LOW_CONF_BAND[1]:
        return "Low confidence — insufficient evidence", True
    conf = abs(p - 0.5) * 2.0
    return ("Higher confidence" if conf >= 0.5 else "Moderate confidence"), False


def uncertainty_panel(p: float, ci=None, label: str = "P(MDD)") -> bool:
    """Show a calibrated probability with a confidence state. Returns flag-for-review."""
    state, flag = confidence_state(p)
    st.metric(label, f"{p:.0%}")
    if ci:
        st.caption(f"Approx. interval: {ci[0]:.0%} – {ci[1]:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    if flag:
        st.error(f"**{state}** → automatically flagged for clinician review.")
    else:
        st.info(f"**{state}**")
    return flag


def contributions_chart(contribs, top: int = 8):
    """Horizontal bar of signed feature contributions (positive → toward MDD)."""
    import pandas as pd

    df = pd.DataFrame(contribs[:top], columns=["feature", "contribution"])
    st.bar_chart(df.set_index("feature"), horizontal=True)
    st.caption("Positive (right) pushes toward **MDD**; negative (left) toward **HC**. "
               "Linear model: coefficient × standardized feature value.")


def hitl_controls(case_id: str):
    """Human-in-the-loop: the tool suggests; the clinician confirms/overrides/escalates."""
    st.markdown("**The clinician decides — the tool only suggests.**")
    c1, c2, c3 = st.columns(3)
    decision = None
    if c1.button("✅ Confirm suggestion", key=f"confirm_{case_id}"):
        decision = "confirmed"
    if c2.button("✏️ Override", key=f"override_{case_id}"):
        decision = "overridden"
    if c3.button("🔬 Request human review", key=f"review_{case_id}"):
        decision = "review"
    if decision:
        st.success(f"Recorded clinician action: **{decision}** (demo — not persisted).")
    return decision


def real_badge():
    st.markdown(":green[**● Real — Layer A model on open EEG data**]")


def illustrative_badge():
    st.markdown(":orange[**● Illustrative — synthetic, not a validated prediction**]")
