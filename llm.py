"""The generation step: ask a language model to translate the document into the
structured plain-language result — or, with no API key, fall back to a cached demo
result so the app (and a demo video) always works.

Provider-agnostic: set ANY ONE of ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY.
Capabilities used here: NLP understanding, summarization, classification (urgency),
and grounded generation over the retrieved resources.
"""

import json
import os

from prompts import SYSTEM_PROMPT, build_user_message


class NoKeyError(Exception):
    """Raised when custom text is submitted but no provider API key is configured."""


# Set MINDBRIDGE-style any-one-key. Force a provider with CLEARSTEP_PROVIDER.
_PROVIDER_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
}


def _detect_provider():
    forced = os.environ.get("CLEARSTEP_PROVIDER")
    order = [forced] if forced else ["anthropic", "openai", "gemini"]
    for name in order:
        for k in _PROVIDER_KEYS.get(name, ()):
            if os.environ.get(k):
                return name, os.environ[k]
    return None, None


def _model_for(provider):
    return os.environ.get("CLEARSTEP_MODEL", _DEFAULT_MODELS[provider])


def analyze(document_text: str, resources_block: str, example_key: str | None = None):
    """Return the structured analysis dict for a document.

    With any provider key: a live model call. Without one: the cached demo result for
    a bundled example, if available. Raises NoKeyError for custom text with no key.
    """
    provider, key = _detect_provider()
    if key:
        result = _analyze_live(provider, key, document_text, resources_block)
        result["mode"] = "live"
        result["provider"] = provider
        return result

    if example_key and example_key in DEMO_RESPONSES:
        result = json.loads(json.dumps(DEMO_RESPONSES[example_key]))  # deep copy
        result["mode"] = "demo"
        return result

    raise NoKeyError(
        "No API key found. Live analysis of custom text needs one of ANTHROPIC_API_KEY, "
        "OPENAI_API_KEY, or GEMINI_API_KEY — or try a built-in example (runs key-free)."
    )


def _analyze_live(provider, key, document_text, resources_block) -> dict:
    """Dispatch the same prompt to whichever provider's key is configured."""
    user = build_user_message(document_text, resources_block)
    model = _model_for(provider)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model, max_tokens=1500, temperature=0.2, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model, temperature=0.2, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content

    elif provider == "gemini":
        import google.generativeai as genai
        # transport="rest" avoids a gRPC "Event loop is closed" error inside Streamlit.
        genai.configure(api_key=key, transport="rest")
        gm = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)
        resp = gm.generate_content(
            user, generation_config={"temperature": 0.2,
                                     "response_mime_type": "application/json"},
        )
        text = resp.text

    else:
        raise NoKeyError(f"Unknown provider: {provider}")

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
    },
    "insurance_denial": {
        "urgency": "routine",
        "urgency_reason": "You have up to 180 days to appeal and there is no emergency — "
        "but this denial can be challenged, and it's worth acting well before the deadline.",
        "summary": "Your insurance said no to paying for 8 therapy sessions, calling them "
        "'not medically necessary.' That is a common denial, and you have the right to "
        "appeal it. If it stands, you could owe $1,240 — but many denials like this get "
        "overturned once the provider sends more documentation.",
        "key_points": [
            "This is a DENIAL you can fight — not a final bill. You have 180 days to "
            "appeal in writing.",
            "The reason given is 'not medically necessary' and that the records didn't "
            "show a lower level of care was tried first.",
            "Your therapist can send extra documentation to support medical necessity — "
            "ask them to help with the appeal.",
            "If a delay would seriously harm the patient's health, you can request an "
            "EXPEDITED (faster) appeal.",
            "If the internal appeal fails, you can ask for an external review by an "
            "Independent Review Organization (IRO)."
        ],
        "checklist": [
            {"step": "Call Member Services (number on the insurance card) and say you "
             "want to file an internal appeal.", "timeframe": "this week",
             "why": "Starts the appeal and confirms exactly what they need."},
            {"step": "Ask your therapist/provider to write a letter of medical necessity "
             "and send their records.", "timeframe": "within a week or two",
             "why": "Provider documentation is what overturns most of these denials."},
            {"step": "Submit the written appeal before the 180-day deadline; keep copies "
             "of everything.", "timeframe": "before the deadline (sooner is better)",
             "why": "Miss the window and you lose the right to appeal."},
            {"step": "If the patient's health can't wait, request an EXPEDITED appeal.",
             "timeframe": "right away if needed", "why": "Expedited appeals are decided "
             "much faster."},
            {"step": "Call 211 or NAMI for help understanding the process or finding "
             "lower-cost care meanwhile.", "timeframe": "as needed",
             "why": "They help families navigate insurance and find affordable options."}
        ],
        "watch_for": [
            "If the teen's mental health gets worse while you wait — don't let the "
            "insurance fight delay care; call 988 or seek help",
            "Any sign of self-harm or crisis — that's an emergency, separate from the "
            "insurance issue"
        ],
        "resource_ids": ["988", "crisis-text", "211", "nami", "findtreatment"]
    }
}
