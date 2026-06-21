"""Generation step: ask Claude to produce the two-audience output (clinician
evidence + family explainer), grounded in the retrieved guidelines and resources.

With an API key it calls Claude live for any patient. Without one, it returns a
cached, realistic result for the three built-in demo patients — so the app and a
demo video always work. The ML signal itself is always real (model_engine), key
or no key.
"""

import json
import os

from prompts import SYSTEM_PROMPT, build_user_message

class NoKeyError(Exception):
    """Raised when a custom patient is analyzed but no provider API key is set."""


# Provider-agnostic: set any ONE of these keys to enable live analysis. Force a
# provider with MINDBRIDGE_PROVIDER, otherwise the first key found wins.
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
    forced = os.environ.get("MINDBRIDGE_PROVIDER")
    order = [forced] if forced else ["anthropic", "openai", "gemini"]
    for name in order:
        for k in _PROVIDER_KEYS.get(name, ()):
            if os.environ.get(k):
                return name, os.environ[k]
    return None, None


def _model_for(provider):
    return os.environ.get("MINDBRIDGE_MODEL", _DEFAULT_MODELS[provider])


def explain(patient_block, signal_block, guidelines_block, resources_block,
            demo_seed=None, note_block=""):
    provider, key = _detect_provider()
    if key:
        out = _explain_live(provider, key, patient_block, signal_block,
                            guidelines_block, resources_block, note_block)
        out["mode"] = "live"
        out["provider"] = provider
        return out
    if demo_seed in DEMO_RESPONSES:
        out = json.loads(json.dumps(DEMO_RESPONSES[demo_seed]))  # deep copy
        out["mode"] = "demo"
        return out
    raise NoKeyError(
        "Live analysis of a custom patient needs an API key — set ANTHROPIC_API_KEY, "
        "OPENAI_API_KEY, or GEMINI_API_KEY. The three demo patients run with no key."
    )


def _extract_json(text):
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"No JSON object in model output:\n{text[:300]}")
    return json.loads(text[s : e + 1])


def _explain_live(provider, key, patient_block, signal_block, guidelines_block,
                  resources_block, note_block=""):
    """Dispatch the same prompt to whichever provider's key is configured."""
    user = build_user_message(patient_block, signal_block, guidelines_block,
                              resources_block, note_block)
    model = _model_for(provider)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model, max_tokens=1800, temperature=0.2, system=SYSTEM_PROMPT,
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
        genai.configure(api_key=key)
        gm = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)
        resp = gm.generate_content(
            user, generation_config={"temperature": 0.2,
                                     "response_mime_type": "application/json"},
        )
        text = resp.text

    else:
        raise NoKeyError(f"Unknown provider: {provider}")

    return _extract_json(text)


# --- Cached demo results for the three built-in patients (keyed by seed) ---------

DEMO_RESPONSES = {
    373: {  # P(MDD)=0.82 — elevated
        "clinician": {
            "signal_summary": "The EEG model returns an elevated signal (P≈0.82 for the "
            "MDD-associated pattern), clearly above the decision threshold but with wide "
            "model uncertainty (LOSO AUC 95% CI 0.88–1.00); treat it as one supportive "
            "data point, not a diagnosis.",
            "evidence_to_consider": [
                {"point": "Confirm with a clinical interview and a validated measure "
                 "(PHQ-A) rather than acting on the signal alone.",
                 "source": "GLAD-PC; APA Clinical Practice Guideline"},
                {"point": "For moderate-severe presentations, evidence-based "
                 "psychotherapy (CBT/IPT) and/or an antidepressant are first-line.",
                 "source": "GLAD-PC: Guidelines for Adolescent Depression in Primary Care"},
                {"point": "If an antidepressant is considered, note the FDA boxed warning "
                 "for suicidality in youth and plan close early monitoring.",
                 "source": "U.S. FDA boxed warning"},
            ],
            "safety_flags": [
                "Screen explicitly for suicidal ideation and plan; arrange safety "
                "planning and means restriction if any risk is present."
            ],
            "caveat": "This is decision support, not a decision — no EEG or hormonal "
            "marker is validated to diagnose or to select treatment for an individual.",
        },
        "family": {
            "summary": "A computer tool looked at the brainwave summary and noticed a "
            "pattern that can sometimes go along with depression. It is NOT sure, and it "
            "is NOT a diagnosis — it's just one clue. What matters most is a full check-in "
            "with your clinician.",
            "what_matters": [
                "This is one clue, not an answer — your clinician decides what it means.",
                "The most helpful next step is a real conversation and a simple "
                "questionnaire (like the PHQ-A) with your care team.",
                "If your teen ever talks about self-harm, get help right away — don't wait."
            ],
            "next_steps": [
                {"step": "Book or confirm the follow-up appointment with your clinician "
                 "or an outpatient provider (FindTreatment.gov can help).",
                 "timeframe": "this week"},
                {"step": "Save 988 in your phone and lock up any medications and firearms.",
                 "timeframe": "today"},
                {"step": "Write down what you've noticed (sleep, mood, appetite) to share "
                 "at the visit.", "timeframe": "before the appointment"},
            ],
            "resource_ids": ["988", "crisis-text", "findtreatment", "nami", "jed", "211"],
        },
    },
    359: {  # P(MDD)=0.45 — borderline / inconclusive
        "clinician": {
            "signal_summary": "The EEG signal is borderline (P≈0.45), inside the model's "
            "low-confidence band (0.40–0.60). It is effectively inconclusive and should "
            "not move the assessment in either direction.",
            "evidence_to_consider": [
                {"point": "Base the assessment on clinical interview and a validated "
                 "measure; the model adds little here.",
                 "source": "APA Clinical Practice Guideline; GLAD-PC"},
                {"point": "Use measurement-based care (PHQ-A) over time to clarify the "
                 "picture rather than a single test.",
                 "source": "GLAD-PC; APA Clinical Practice Guideline"},
            ],
            "safety_flags": ["Still screen for suicide risk regardless of the signal."],
            "caveat": "An inconclusive model output is not reassurance — clinical "
            "judgment governs.",
        },
        "family": {
            "summary": "The computer tool could not tell much from the brainwave summary "
            "this time — its result was right in the middle. That's okay. It just means "
            "the tool isn't the part that matters here; your clinician's check-in is.",
            "what_matters": [
                "An 'unsure' result is normal — the tool is only a small helper.",
                "Your clinician's questions and your own observations matter most.",
                "Keep watching for changes and stay in touch with your care team."
            ],
            "next_steps": [
                {"step": "Keep the appointment with your clinician and share what you've "
                 "noticed.", "timeframe": "this week"},
                {"step": "Save 988 and the Crisis Text Line in your phone, just in case.",
                 "timeframe": "today"},
            ],
            "resource_ids": ["988", "crisis-text", "findtreatment", "nami", "teenline"],
        },
    },
    487: {  # P(MDD)=0.17 — low signal
        "clinician": {
            "signal_summary": "The EEG signal is low (P≈0.17 for the MDD-associated "
            "pattern). Importantly, a low signal does NOT rule out depression — the model "
            "detects a group-level pattern, not the disorder.",
            "evidence_to_consider": [
                {"point": "Do not treat a low signal as reassurance; continue full "
                 "clinical assessment if symptoms are present.",
                 "source": "Consensus of current evidence (no biomarker is diagnostic)"},
                {"point": "If depression is diagnosed clinically, psychotherapy (CBT/IPT) "
                 "is first-line for adolescents.",
                 "source": "NICE NG134; AACAP practice parameters"},
            ],
            "safety_flags": ["Screen for suicide risk on clinical grounds, not on the signal."],
            "caveat": "Absence of a biomarker signal is not evidence of absence of "
            "depression; this is decision support only.",
        },
        "family": {
            "summary": "The computer tool did not see the brainwave pattern it looks for. "
            "That is NOT a clean bill of health — the tool can miss things, and only your "
            "clinician can say what's really going on. If you're worried, that worry still "
            "matters.",
            "what_matters": [
                "A 'low' result does not mean everything is fine.",
                "Trust what you see day to day, and tell your clinician about it.",
                "You can still ask for a full evaluation and support."
            ],
            "next_steps": [
                {"step": "If you have concerns, schedule an evaluation with a provider "
                 "(FindTreatment.gov or call 211).", "timeframe": "this week"},
                {"step": "Keep 988 saved in case things change.", "timeframe": "today"},
            ],
            "resource_ids": ["988", "crisis-text", "findtreatment", "211", "jed"],
        },
    },
}
