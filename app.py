"""ClearStep — turn a confusing mental-health document into plain language,
a clear checklist, and trusted next steps.

Built for the USAII Global AI Hackathon 2026 (High School track,
Challenge 1: "Help is Hard to Find").

Flow:  paste a document  ->  retrieve public resources (RAG)  ->  Claude
explains + classifies urgency + builds a checklist  ->  the reader gets clear,
verifiable next steps, with a crisis line always on screen.
"""

from pathlib import Path

import streamlit as st

from llm import NoKeyError, analyze
from rag import ResourceRetriever

EXAMPLES = {
    "ER discharge after a mental-health crisis (Jordan, 15)": "er_discharge",
    "School counselor referral letter": "school_referral",
}
EXAMPLE_DIR = Path(__file__).parent / "examples"

URGENCY_STYLE = {
    "emergency": ("🚨", "#fee2e2", "#991b1b", "Emergency — act now"),
    "urgent": ("⏳", "#fef3c7", "#92400e", "Time-sensitive — act soon"),
    "routine": ("✅", "#dcfce7", "#166534", "Important — no tight deadline"),
}


@st.cache_resource
def get_retriever():
    return ResourceRetriever()


def load_example(key: str) -> str:
    return (EXAMPLE_DIR / f"{key}.txt").read_text(encoding="utf-8")


def crisis_banner():
    st.markdown(
        """
        <div style="background:#991b1b;color:white;padding:12px 16px;border-radius:8px;
        font-size:0.95rem;line-height:1.4;">
        <b>If you or someone else is in danger right now,</b> call or text
        <b>988</b> (Suicide &amp; Crisis Lifeline) or text <b>HOME to 741741</b>.
        If someone is unsafe, call 911 or go to the nearest ER.
        </div>
        """,
        unsafe_allow_html=True,
    )


def resource_card(r):
    st.markdown(
        f"""
        <div style="border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;
        margin-bottom:8px;background:#ffffff;">
          <div style="font-weight:600;">{r['name']}</div>
          <div style="margin:4px 0;">{r['description']}</div>
          <div style="font-size:0.9rem;"><b>Contact:</b> {r['contact']} &nbsp;·&nbsp;
          <a href="{r['url']}" target="_blank">{r['url']}</a></div>
          <div style="font-size:0.8rem;color:#64748b;">Source: {r['source']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result, retriever, document_text):
    # --- Responsible AI: crisis line is ALWAYS first, whatever the AI decided ---
    crisis_banner()
    st.write("")

    icon, bg, fg, label = URGENCY_STYLE.get(result["urgency"], URGENCY_STYLE["routine"])
    st.markdown(
        f"""<div style="background:{bg};color:{fg};padding:14px 16px;border-radius:8px;
        font-weight:600;">{icon} {label}</div>""",
        unsafe_allow_html=True,
    )
    st.caption(f"Why: {result['urgency_reason']}")

    st.subheader("What this says, in plain language")
    st.write(result["summary"])
    with st.expander("Show the original document (nothing is hidden)"):
        st.text(document_text)

    st.subheader("What matters most")
    for point in result["key_points"]:
        st.markdown(f"- {point}")

    st.subheader("Your next steps")
    for i, item in enumerate(result["checklist"], 1):
        st.checkbox(
            f"**{item['step']}**  ·  _{item['timeframe']}_",
            key=f"step_{i}",
            help=item.get("why", ""),
        )

    if result.get("watch_for"):
        st.subheader("Get emergency help right away if you notice")
        for w in result["watch_for"]:
            st.markdown(f"- {w}")

    st.subheader("Resources that can help")
    shown_ids = set()
    for r in retriever.always_show():  # 988 + Crisis Text Line, always
        resource_card(r)
        shown_ids.add(r["id"])
    for r in retriever.by_ids(result.get("resource_ids", [])):
        if r["id"] not in shown_ids:
            resource_card(r)
            shown_ids.add(r["id"])

    # --- Human-in-the-loop handoff ---
    st.divider()
    st.subheader("Talk to a real person")
    st.markdown(
        "ClearStep explains and organizes — it does **not** decide if this is an "
        "emergency and does **not** give medical advice. For the decisions that "
        "matter, reach a trained human:"
    )
    c1, c2, c3 = st.columns(3)
    c1.link_button("📞 Call/text 988", "https://988lifeline.org", use_container_width=True)
    c2.link_button("💬 Crisis Text Line", "https://www.crisistextline.org", use_container_width=True)
    c3.link_button("🔎 Find a provider", "https://findtreatment.gov", use_container_width=True)

    if st.button("⚠️ This doesn't look right — flag for a human to review"):
        st.warning(
            "Thanks for flagging. Please don't rely on this summary — confirm the "
            "details with your provider, or call 988 to talk it through with a person."
        )


def main():
    st.set_page_config(page_title="ClearStep", page_icon="🧭", layout="centered")

    st.title("🧭 ClearStep")
    st.markdown(
        "**Confusing mental-health letter? Paste it below.** ClearStep explains it in "
        "plain language, tells you what matters most, and gives you a clear checklist "
        "and trusted places to get help."
    )
    st.info(
        "ℹ️ ClearStep is **not** a doctor and gives **no** medical advice. It only "
        "explains a document a clinician already wrote and points you to human help. "
        "Always confirm with your provider.",
        icon="ℹ️",
    )

    retriever = get_retriever()

    st.markdown("##### 1. Start with an example, or paste your own")
    example_label = st.selectbox(
        "Load an example document", ["— none —", *EXAMPLES.keys()]
    )
    example_key = EXAMPLES.get(example_label)
    default_text = load_example(example_key) if example_key else ""

    document_text = st.text_area(
        "Paste the confusing document here",
        value=default_text,
        height=240,
        placeholder="Paste a discharge summary, school letter, insurance notice...",
    )

    go = st.button("Explain this for me", type="primary", use_container_width=True)

    if go:
        if not document_text.strip():
            st.error("Please paste a document or pick an example first.")
            return

        # If the text matches a loaded example, allow demo mode (no API key needed).
        active_example = example_key if document_text.strip() == default_text.strip() else None

        with st.spinner("Reading the document and finding the right help..."):
            resources = retriever.retrieve(document_text, top_k=5)
            try:
                result = analyze(
                    document_text,
                    ResourceRetriever.to_block(resources),
                    example_key=active_example,
                )
            except NoKeyError as e:
                st.warning(str(e))
                return
            except Exception as e:  # surface model/parse errors plainly
                st.error(f"Something went wrong analyzing the document: {e}")
                return

        if result.get("mode") == "demo":
            st.caption("Running in demo mode (cached AI result — no API key set).")
        st.divider()
        render_result(result, retriever, document_text)


if __name__ == "__main__":
    main()
