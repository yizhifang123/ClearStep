"""The generation step: ask Claude to translate the document into the structured
plain-language result — or, with no API key, fall back to a cached demo result
so the app (and a demo video) always works.

Capabilities used here: NLP understanding, summarization, classification
(urgency), and grounded generation over the retrieved resources.
"""

import json
import os

from prompts import SYSTEM_PROMPT, build_user_message

DEFAULT_MODEL = os.environ.get("CLEARSTEP_MODEL", "claude-sonnet-4-6")


class NoKeyError(Exception):
    """Raised when custom text is submitted but no API key is configured."""


def _api_key():
    return os.environ.get("ANTHROPIC_API_KEY")


def analyze(document_text: str, resources_block: str, example_key: str | None = None):
    """Return the structured analysis dict for a document.

    With an API key: a live Claude call. Without one: the cached demo result for a
    bundled example, if available. Raises NoKeyError for custom text with no key.
    """
    if _api_key():
        result = _analyze_live(document_text, resources_block)
        result["mode"] = "live"
        return result

    if example_key and example_key in DEMO_RESPONSES:
        result = json.loads(json.dumps(DEMO_RESPONSES[example_key]))  # deep copy
        result["mode"] = "demo"
        return result

    raise NoKeyError(
        "No ANTHROPIC_API_KEY found. Live analysis of custom text needs a key — "
        "or try one of the built-in example documents, which run in demo mode."
    )


def _analyze_live(document_text: str, resources_block: str) -> dict:
    """Call Claude and parse the structured JSON it returns."""
    import anthropic

    client = anthropic.Anthropic(api_key=_api_key())
    message = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1500,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_message(document_text, resources_block),
            }
        ],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    """Pull the JSON object out of the model's reply, tolerating stray prose."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text[:300]}")
    return json.loads(text[start : end + 1])


# --- Cached demo results for the bundled examples -------------------------------
# These let the app run convincingly with no API key (e.g. while recording a demo).
# They are realistic outputs of the same prompt, not hand-waved placeholders.

DEMO_RESPONSES = {
    "er_discharge": {
        "urgency": "urgent",
        "urgency_reason": "The doctors say to start outpatient care within 7 days, "
        "and there are safety steps to take right away — but this is not an active "
        "emergency.",
        "summary": "Jordan was seen in the ER for an anxiety crisis and some thoughts "
        "of self-harm, which the doctors say had passed by the time of discharge. "
        "Jordan is safe to go home. The paper asks the family to start regular mental-"
        "health care soon, keep taking the current medicines, and make the home safer.",
        "key_points": [
            "Jordan is okay to be home, but needs to start counseling/outpatient care "
            "within 7 days — the clock has already started.",
            "Keep taking sertraline 50 mg once a day. Hydroxyzine can be used for "
            "anxiety as needed; it can cause drowsiness.",
            "The doctors asked the family to lock up or remove any guns and to secure "
            "all medicines at home — this is a safety step, not optional.",
            "A referral was placed to the County Behavioral Health Access Line, but the "
            "FAMILY has to call to actually schedule the first appointment."
        ],
        "checklist": [
            {"step": "Call the County Behavioral Health Access Line (or use "
             "FindTreatment.gov) to schedule the first outpatient appointment.",
             "timeframe": "within 7 days (call tomorrow)", "why": "The doctors set a "
             "7-day window to start care."},
            {"step": "Lock up or remove firearms, and put all medicines somewhere "
             "secure.", "timeframe": "today", "why": "A safety step the ER specifically "
             "asked the family to do."},
            {"step": "Keep giving sertraline 50 mg once daily; use hydroxyzine only as "
             "the label says.", "timeframe": "ongoing", "why": "The ER said to continue "
             "current medicines."},
            {"step": "Schedule a primary care (regular doctor) follow-up.",
             "timeframe": "within 1-2 weeks", "why": "Listed as a follow-up step."},
            {"step": "Save 988 in your phone and keep the safety plan somewhere easy to "
             "find.", "timeframe": "today", "why": "So help is one tap away if things "
             "get worse."}
        ],
        "watch_for": [
            "Talk of wanting to die, a plan to self-harm, or giving things away",
            "Sudden worsening mood, agitation, or not being able to stay safe",
            "Any sign Jordan cannot keep themselves safe — call 988 or go to the ER"
        ],
        "resource_ids": ["988", "crisis-text", "findtreatment", "samhsa-helpline",
                          "211", "jed"]
    },
    "school_referral": {
        "urgency": "routine",
        "urgency_reason": "This is a recommendation to set up an evaluation — important, "
        "but there is no tight deadline and no sign of immediate danger.",
        "summary": "The school counselor noticed changes in your child's mood, "
        "attendance, and engagement, and is recommending an outside mental-health "
        "evaluation. The school can't provide treatment itself, so it's pointing you to "
        "community providers. You'll need to sign a release form before the school can "
        "talk to any outside provider.",
        "key_points": [
            "This is a referral, not a diagnosis — the school is suggesting you get a "
            "professional evaluation.",
            "You'll need to sign a 'Release of Information' for the school and a provider "
            "to coordinate.",
            "You can ask the school about support and accommodations, including a 504 "
            "plan.",
            "If you ever think your child is in immediate danger, don't wait for an "
            "appointment — get emergency help."
        ],
        "checklist": [
            {"step": "Use FindTreatment.gov or call 211 to find a mental-health "
             "provider that takes your insurance.", "timeframe": "this week",
             "why": "The school recommends an outside evaluation."},
            {"step": "Ask the counseling office for the provider list and the Student "
             "Assistance Program (SAP) contact.", "timeframe": "this week",
             "why": "They offered these and they save you time."},
            {"step": "Ask the school about a 504 plan or accommodations.",
             "timeframe": "when you meet", "why": "It can support your child at school "
             "during this time."},
            {"step": "Save 988 and Crisis Text Line in your phone.", "timeframe": "today",
             "why": "So help is ready if things change quickly."}
        ],
        "watch_for": [
            "Talk of self-harm or hopelessness",
            "A sudden, sharp drop in mood, sleep, or eating",
            "Withdrawing from friends and activities much more than usual"
        ],
        "resource_ids": ["988", "crisis-text", "findtreatment", "211", "nami", "jed"]
    }
}
