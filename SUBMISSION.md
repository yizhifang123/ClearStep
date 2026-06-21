# ClearStep — Devpost Submission Copy

Copy-paste each block into the matching Devpost field. Character counts are noted;
fields with hard limits are kept under them.

---

## Qualifier Approval Code
```
HS26-86A98BB5
```

## Track & Challenge
- **Track:** High School (Grades 9–12)
- **Challenge:** Challenge 1 — *Help is Hard to Find* · Direction A: Crisis-to-Action Translator

## Project Title
```
ClearStep
```

## Tagline / Elevator pitch (≤80 chars for tagline)
**Tagline (≤80):**
```
Turns confusing mental-health letters into plain language and clear steps
```
**Elevator pitch (1–2 sentences):**
```
A 15-year-old leaves the ER after a mental-health crisis with a page of jargon and a 7-day deadline nobody explained. ClearStep turns that confusing letter into plain language, an urgency flag, a clear checklist, and trusted human help — with a crisis line always one tap away.
```

---

## Project Description (the "About the Project" field)

**What it does & why it matters**
Every community has mental-health support — but families miss it because the
information is scattered, the language is clinical, and it lands in the worst moment to
read carefully. Meet **Jordan, 15**, just discharged from the ER after a mental-health
crisis. The paperwork says *"establish outpatient behavioral health within 7 days,"*
lists medications, and buries the warning signs in jargon. Jordan's mom reads it twice
and still doesn't know what to **do** — and the clock is already running.

**ClearStep** turns that confusing document into clarity and action. Paste the letter
and you get: a plain-language explanation (~6th-grade reading level), an urgency flag
(emergency / time-sensitive / routine), the few things that matter most, a step-by-step
checklist with timeframes, and verified resources (988, Crisis Text Line, 211, SAMHSA,
NAMI…) each with a source link — plus a crisis line on every screen and a one-tap
handoff to a real person.

**Why AI, not a web search:** a search engine can't read *your specific letter*.
ClearStep understands *this* document's jargon, judges *its* urgency, rewrites *it* in
plain language, and maps it to the right help — per-document language understanding,
summarization, classification, and retrieval that a static checklist or search bar
cannot do.

**How we built it**
A Streamlit app with a clean two-step AI pipeline. (1) **Retrieval (RAG):** scikit-learn
TF-IDF cosine match over a curated, *public* resource directory (`resources.json`) —
never over private data — so answers are grounded in real, linked services, with crisis
lines force-included as a safety floor. (2) **Generation:** Anthropic Claude reads the
document, classifies urgency, summarizes at a 6th-grade level, and builds the checklist,
constrained to *explain, not advise* and to reference only retrieved resources. The app
also ships a **demo mode** with cached AI results so it runs (and demos) with no API key.

**Challenges**
Keeping the AI from "helping too much." The hard part of a medical-adjacent tool is
making the model *explain an instruction a clinician already wrote* instead of inventing
advice. We solved it with a tightly constrained prompt (explain-don't-advise, only
retrieved resources, no invented numbers) and UI guardrails that don't depend on the
model behaving — the 988 card and the original-text view are always shown.

**Accomplishments we're proud of**
A genuinely safe design: the crisis line shows regardless of what the AI decides, the
original document is never hidden, and the AI explicitly does *not* make the safety call.
It's helpful and humble at the same time.

**What we learned**
The most useful AI here isn't the smartest answer — it's the *clearest* one delivered
with the right guardrails. Responsible design (where the human stays in control) is a
feature users can feel, not a disclaimer.

**What's next**
Local resources by ZIP via the 211 API, multi-language output, photo/PDF upload with
OCR, and a school-counselor mode for sending plain-language explainers to families.

**Built with:** Python · Streamlit · Anthropic Claude API · scikit-learn (TF-IDF / RAG) ·
Claude Code (AI coding assistance, disclosed)

---

## AI Architecture Explanation (≤600 chars)
```
Inputs: a confusing mental-health document (ER discharge, school referral, insurance notice) pasted as text. AI capabilities: NLP + summarization + urgency classification + retrieval (RAG) + generative AI. Processing: scikit-learn TF-IDF retrieves matching services from a curated PUBLIC resource directory; Claude then explains the document at a 6th-grade level, rates urgency, and builds a checklist grounded only in those resources. Outputs: plain-language summary, "what matters most," an action checklist with timeframes, verified resources with links, and an always-on 988 crisis card.
```

## Human-in-the-Loop Design (≤500 chars)
```
ClearStep does NOT decide whether a situation is a true emergency, and it gives NO medical advice. It only explains instructions a clinician already wrote and shows an urgency signal. The actual safety and treatment decisions route to a trained human — 988, Crisis Text Line, or the family's provider — reachable in one tap on every screen. Why: misjudging a mental-health crisis can be fatal, and only a trained person can weigh context (tone, history, what's unsaid) that an AI cannot see.
```

## Responsible AI Guardrail (≤500 chars)
```
Risk: the AI under-rates urgency or drops a critical instruction (e.g., a medication warning), so a stressed reader takes the wrong action in a crisis. Mitigation: the 988 crisis line is shown on EVERY result regardless of the AI's urgency rating; the original document is always displayed beside the summary, so nothing is hidden; and every resource carries a source link the reader can verify. The guardrails live in the UI, so they hold even if the model is wrong.
```

## Tools Used (≤800 chars)
```
- Anthropic Claude API (model: claude-sonnet-4-6) — paid (free trial credits available). Used for NLP understanding, summarization, urgency classification, and plain-language generation.
- scikit-learn (free, open-source) — TF-IDF retrieval over the public resource directory (the RAG step).
- Streamlit (free, open-source) — the web app UI.
- Python 3 standard library (free).
- AI coding assistance: Claude Code (disclosed) — used to scaffold code and refine copy.

No paid tool is required to run the project: a built-in demo mode uses cached AI results, so the full app works for free.
```

## Data Disclosure (≤800 chars)
```
- Public resource directory (curated by us; all official public services), each with its real contact and source link: 988 Suicide & Crisis Lifeline, Crisis Text Line, SAMHSA National Helpline, FindTreatment.gov, 211 (United Way), NAMI, The Trevor Project, Teen Line, The Jed Foundation, Trans Lifeline. Stored in resources.json.
- Example documents: SYNTHETIC. We wrote realistic mock letters (an ER discharge summary and a school counselor referral). No real patient data is used.
- No sensitive or personal data is collected or stored. Text the user pastes is sent to the Claude API only to generate the explanation and is not retained by the app.
```

## Demo Materials
- **Pitch video (3–5 min):** _[paste YouTube/Loom link]_ — see `DEMO_SCRIPT.md` for the script.
- **Working demo / walkthrough:** GitHub repo (this README + run instructions). Optionally a deployed Streamlit link.
