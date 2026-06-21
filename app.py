"""ClearStep — turn any confusing mental-health document into plain language, a
clear checklist, and TRUSTWORTHY next steps — while being honest about what it
does not know, and refusing to ever diagnose.

Built for the USAII Global AI Hackathon 2026 (High School track,
Challenge 1: "Help is Hard to Find").

Flow:  paste a confusing document  ->  retrieve public resources (RAG)  ->  a
language model explains it in plain language + builds a checklist  ->  the reader
gets clear, verifiable next steps, a per-resource trust ribbon, and an always-on
crisis line. ClearStep never diagnoses; it routes the real decisions to humans.
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # read clearstep/.env so an API key there is picked up automatically

from llm import NoKeyError, analyze  # noqa: E402
from rag import ResourceRetriever  # noqa: E402

EXAMPLES = {
    "ER discharge after a mental-health crisis": "er_discharge",
    "School counselor referral letter": "school_referral",
    "Insurance denial for therapy": "insurance_denial",
}
EXAMPLE_DIR = Path(__file__).parent / "examples"

URGENCY_STYLE = {
    "emergency": ("#fee2e2", "#991b1b", "Emergency — act now"),
    "urgent": ("#fef3c7", "#92400e", "Time-sensitive — act soon"),
    "routine": ("#dcfce7", "#166534", "Important — no tight deadline"),
}


@st.cache_resource
def get_retriever():
    return ResourceRetriever()


def load_example(key: str) -> str:
    return (EXAMPLE_DIR / f"{key}.txt").read_text(encoding="utf-8")


def never_diagnose_banner():
    st.markdown(
        """<div style="background:#eef2ff;border-left:4px solid #4f46e5;padding:10px 14px;
        border-radius:6px;font-size:0.92rem;line-height:1.45;">
        <b>ClearStep will never tell you if you or your child has depression</b>, and gives
        <b>no</b> medical advice. It makes confusing information clear and points you to
        people who can help. The real decisions stay with you and a trained human.</div>""",
        unsafe_allow_html=True,
    )


def crisis_banner():
    st.markdown(
        """<div style="background:#991b1b;color:white;padding:11px 15px;border-radius:8px;
        line-height:1.4;"><b>If you or someone else is in danger right now,</b> call or text
        <b>988</b> (Suicide &amp; Crisis Lifeline) or text <b>HOME to 741741</b>. If someone
        is unsafe, call 911 or go to the nearest ER.</div>""",
        unsafe_allow_html=True,
    )


def trust_ribbon(r):
    """The signature: tell the reader WHO runs each resource and whether it's free,
    confidential, and available now — because 'help is hard to find' is really
    'which of these can I trust?'"""
    free = "Free" if r.get("free") else "May have a cost"
    conf = "Confidential" if r.get("confidential") else "Not confidential"
    parts = [f"Run by {r.get('run_by', r.get('source', '—'))}", free, conf,
             r.get("availability", "")]
    chips = "  ·  ".join(p for p in parts if p)
    return (f"<div style='font-size:0.8rem;color:#475569;background:#f1f5f9;"
            f"border-radius:6px;padding:4px 8px;margin-top:6px;'>{chips}</div>")


def resource_card(r):
    st.markdown(
        f"""<div style="border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;
        margin-bottom:8px;background:#ffffff;">
          <div style="font-weight:600;">{r['name']}</div>
          <div style="margin:4px 0;">{r['description']}</div>
          <div style="font-size:0.9rem;"><b>Contact:</b> {r['contact']} &nbsp;·&nbsp;
          <a href="{r['url']}" target="_blank">{r['url']}</a></div>
          {trust_ribbon(r)}
        </div>""",
        unsafe_allow_html=True,
    )


def honesty_stamp():
    st.markdown(
        """<div style="border:1px dashed #94a3b8;border-radius:8px;padding:8px 12px;
        background:#f8fafc;font-size:0.88rem;color:#334155;">
        <b>What we don't know:</b> this is <b>general information</b>, not advice about your
        specific situation. ClearStep can be wrong or miss something — always confirm the
        details with a real person before acting.</div>""",
        unsafe_allow_html=True,
    )


def about_our_ai():
    st.markdown(
        """<div style="border:2px solid #4f46e5;border-radius:10px;padding:16px 18px;
        background:#eef2ff;">
        <div style="font-weight:700;font-size:1.05rem;color:#3730a3;">
        We built a real depression-detection AI — then refused to use it.</div>
        <div style="margin-top:8px;line-height:1.5;">
        Our team actually trained a machine-learning model that detects depression from
        brain-wave (EEG) data. On public research data it reached about <b>95% accuracy</b>
        (ROC-AUC 0.95, validated leave-one-subject-out, p&lt;0.001). Then we
        <b>deliberately left it out of ClearStep.</b><br><br>
        Because no brain scan or algorithm can tell a scared parent what their kid needs —
        and a tool that announces <i>"your child is 82% likely depressed"</i> causes harm,
        not help. <b>The most responsible thing our AI does is refuse to guess about your
        family.</b> That restraint is ClearStep.</div></div>""",
        unsafe_allow_html=True,
    )


def build_takehome(result, resources):
    """A plain-text sheet the family can keep or print."""
    out = ["ClearStep — Your take-home summary", "=" * 36, "",
           "WHAT THIS SAYS, IN PLAIN LANGUAGE", result["summary"], "",
           "WHAT MATTERS MOST"]
    out += [f"- {p}" for p in result["key_points"]]
    out += ["", "YOUR NEXT STEPS"]
    out += [f"[ ] {s['step']}  ({s['timeframe']})" for s in result["checklist"]]
    if result.get("watch_for"):
        out += ["", "GET EMERGENCY HELP RIGHT AWAY IF YOU NOTICE"]
        out += [f"- {w}" for w in result["watch_for"]]
    out += ["", "WHERE TO GET HELP"]
    for r in resources:
        out += [f"- {r['name']}: {r['contact']}  ({r.get('availability','')})  {r['url']}"]
    out += ["", "If you or someone else is in danger now, call or text 988, or call 911.", "",
            "This is general information, not medical advice. ClearStep never diagnoses.",
            "Confirm details with a real person. Generated by ClearStep."]
    return "\n".join(out)


def render_result(result, retriever, document_text):
    crisis_banner()
    st.write("")
    honesty_stamp()
    st.write("")

    bg, fg, label = URGENCY_STYLE.get(result["urgency"], URGENCY_STYLE["routine"])
    st.markdown(
        f"""<div style="background:{bg};color:{fg};padding:13px 16px;border-radius:8px;
        font-weight:600;">{label}</div>""",
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
        st.checkbox(f"**{item['step']}**  ·  _{item['timeframe']}_",
                    key=f"step_{i}", help=item.get("why", ""))

    if result.get("watch_for"):
        st.subheader("Get emergency help right away if you notice")
        for w in result["watch_for"]:
            st.markdown(f"- {w}")

    st.subheader("Resources you can trust")
    shown_ids, shown = set(), []
    for r in retriever.always_show():
        resource_card(r); shown_ids.add(r["id"]); shown.append(r)
    for r in retriever.by_ids(result.get("resource_ids", [])):
        if r["id"] not in shown_ids:
            resource_card(r); shown_ids.add(r["id"]); shown.append(r)

    st.download_button("Download a take-home summary",
                       build_takehome(result, shown),
                       file_name="clearstep_summary.txt", use_container_width=True)

    st.divider()
    st.subheader("Talk to a real person")
    st.markdown("ClearStep explains and organizes — it does **not** decide if this is an "
                "emergency and does **not** give medical advice. For the decisions that "
                "matter, reach a trained human:")
    c1, c2, c3 = st.columns(3)
    c1.link_button("Call or text 988", "https://988lifeline.org", use_container_width=True)
    c2.link_button("Crisis Text Line", "https://www.crisistextline.org", use_container_width=True)
    c3.link_button("Find a provider", "https://findtreatment.gov", use_container_width=True)


def main():
    st.set_page_config(page_title="ClearStep", layout="centered")

    st.title("ClearStep")
    st.markdown(
        "**Confusing mental-health letter, bill, or form? Paste it below.** ClearStep "
        "explains it in plain language, tells you what matters most, gives you a clear "
        "checklist, and points you to help you can **trust**."
    )
    never_diagnose_banner()

    retriever = get_retriever()

    st.markdown("##### 1. Start with an example, or paste your own")
    example_label = st.selectbox("Load an example document", ["— none —", *EXAMPLES.keys()])
    example_key = EXAMPLES.get(example_label)
    default_text = load_example(example_key) if example_key else ""

    document_text = st.text_area(
        "Paste the confusing document here", value=default_text, height=220,
        key=f"doc_{example_key}",
        placeholder="Paste a discharge summary, school letter, insurance denial...",
    )

    if st.button("Explain this for me", type="primary", use_container_width=True):
        if not document_text.strip():
            st.error("Please paste a document or pick an example first.")
        else:
            active = example_key if document_text.strip() == default_text.strip() else None
            with st.spinner("Reading the document and finding help you can trust..."):
                resources = retriever.retrieve(document_text, top_k=5)
                try:
                    result = analyze(document_text, ResourceRetriever.to_block(resources),
                                     example_key=active)
                except NoKeyError as e:
                    st.warning(str(e)); result = None
                except Exception as e:
                    st.error(f"Something went wrong analyzing the document: {e}"); result = None
            if result is not None:
                # Persist so later clicks (download, checkboxes) don't blank the page.
                st.session_state.result = result
                st.session_state.doc = document_text

    if st.session_state.get("result"):
        if st.session_state.result.get("mode") == "demo":
            st.caption("Demo mode: cached AI result (no API key set).")
        st.divider()
        render_result(st.session_state.result, retriever, st.session_state.doc)

    st.divider()
    about_our_ai()


if __name__ == "__main__":
    main()
