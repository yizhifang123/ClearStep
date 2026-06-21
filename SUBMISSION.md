# ClearStep — Devpost Submission Copy

Paste each block into the matching Devpost field. Char-limited fields are kept under limit.
NOTE: the Elevator pitch & About open with a representative family; swap in the team's real
personal story (the video cold-open) if you want them identical.

---

## Qualifier Approval Code
```
HS26-86A98BB5
```
*(Double-check against your original email before submitting.)*

## Track & Challenge
- **Track:** High School (Grades 9–12)
- **Challenge:** Challenge 1 — *Help is Hard to Find* · Direction A (turn confusing info into plain language + a checklist + clear next steps)

## Project Title
```
ClearStep
```

## Tagline (≤80 chars)
```
Turns confusing mental-health paperwork into plain, trustworthy next steps
```

## Elevator pitch
```
A parent holds a letter about their teen's mental health — an insurance denial, an ER discharge, a school referral — full of jargon and deadlines nobody explained. ClearStep turns it into plain language, a clear checklist, and resources they can actually trust. And unlike other tools, it flatly refuses to diagnose their child. The most responsible thing our AI does is know what NOT to do.
```

---

## Project Description (the "About the Project" field)

**Who it's for & why it matters**
Help usually *exists* — families miss it because the information is scattered, the language
is built for specialists, and it arrives in the worst moment to read carefully. Picture a
parent at the kitchen table with a letter about their 15-year-old: an insurance denial for
therapy, or an ER discharge that says *"establish outpatient behavioral health within 7
days."* They read it twice and still don't know what to **do** — or which phone number on
the page they can trust. The clock is already running.

**What ClearStep does.** Paste any confusing mental-health document and you get: a
plain-language explanation (~6th-grade reading level), a flag for what's time-sensitive
(like an appeal deadline), the few things that matter most, a step-by-step checklist, and
matched public resources — each carrying a **trust ribbon** showing *who runs it, whether
it's free, whether it's confidential, and when it's available.* A crisis line is on every
screen, and you can download a take-home summary.

**Our signature — honest AI.** To prove we could, our team trained a real machine-learning
model that detects depression from brain-wave (EEG) data, the rigorous way (leave-one-
subject-out cross-validation, permutation-tested): about **0.95 ROC-AUC.** But that was on
**just 58 adults from one public dataset** — no clinical test — so we **deliberately left it
out of ClearStep.** That refusal is a real feature, not a slogan: ask ClearStep to judge a
person ("does my kid have depression?") and a built-in gate **visibly stops and routes you
to a human** instead of answering. ClearStep **never diagnoses**, stamps every answer
*"general information, not advice about your specific situation,"* and shows you the benched
model's own results as the reason it won't guess. The most responsible thing our AI does is
know when *not* to answer.

**Why AI, not a web search:** a search engine can't read *your* letter. ClearStep
understands *this* document's jargon, judges *its* urgency, rewrites *it* in plain language,
and tells you *which* resources to trust for *your* situation — per-document understanding,
summarization, classification, and retrieval a search bar can't do.

**How we built it.** A Streamlit app. (1) **Retrieval (RAG):** scikit-learn TF-IDF over a
curated *public* resource directory — never private data — with crisis lines force-included
as a safety floor. (2) **Generation:** Google Gemini (free tier) explains the document,
flags urgency, and builds the checklist, constrained to *explain, not advise* and to cite
only retrieved resources. A demo mode with cached results runs the whole app with no API key.

**Challenges.** The hard part of a mental-health tool is making it *helpful and humble at
once.* Our answer was to put the guardrails in the interface (so they hold even if the model
is wrong) and to make restraint the product — including benching our own model.

**Accomplishments we're proud of.** A real, validated ML model *and* the discipline to
refuse to use it; the refusal turned into a working safety gate; a trust ribbon that fights
misinformation. Every resource ClearStep shows is one of **10 hand-verified public services**
with checked contact info — the AI is constrained to cite only these, so it has surfaced
**zero invented or unverifiable resources.**

**What we learned.** The most useful AI here isn't the most confident answer — it's the
clearest one, delivered with the human in control.

**What's next.** Local resources by ZIP (211 API), multi-language output, and a counselor
mode that sends families a plain-language explainer.

**Built with:** Python · Streamlit · Google Gemini API (free tier) · scikit-learn (TF-IDF /
RAG) · Claude Code (AI coding assistance, disclosed)

---

## AI Architecture Explanation (≤600 chars)
```
Inputs: any confusing mental-health document — ER discharge, school letter, insurance denial — pasted as text. AI capabilities: NLP + summarization + urgency classification + retrieval (RAG) + generative AI. Processing: scikit-learn TF-IDF retrieves matching public services from a curated directory; Google Gemini explains the document in plain language, flags deadlines, and builds a checklist grounded only in those resources. Outputs: a plain-language summary, "what matters most," an action checklist, resources with a trust ribbon (who runs it, free?, confidential?), and an always-on 988 line.
```

## Human-in-the-Loop Design (≤500 chars)
```
The decision ClearStep refuses to make: whether you or your child has a condition, how serious it is, or what to do about it. A built-in gate detects these questions and visibly STOPS — it shows why and hands off to a human (988, Crisis Text Line, your provider), one tap away. It never diagnoses or gives medical advice; it only explains a document someone else wrote. Why: misjudging a young person's mental health can be life-or-death, and only a person can weigh what an AI can't see.
```

## Responsible AI Guardrail (≤500 chars)
```
Risk: a family over-trusts a confident AI and treats a guess about their child as fact. In-app design choices: (1) a "Should this AI decide?" gate — ask it to judge a person and it visibly REFUSES, routing to a human; (2) every result stamped "general info, not your situation"; (3) the AI cites only our verified resource directory (zero invented help); (4) 988 always on screen. We even trained a real EEG model (0.95 AUC, 58 adults) and benched it — no small-dataset score is a clinical test.
```

## Tools Used (≤800 chars)
```
- Google Gemini API (model: gemini-2.5-flash) — FREE tier. Plain-language generation, summarization, urgency classification. The app is provider-agnostic: one interface also supports Anthropic Claude and OpenAI.
- scikit-learn (free, open-source) — TF-IDF retrieval over the public resource directory (the RAG step).
- Streamlit + python-dotenv (free, open-source) — web app and config.
- AI coding assistance: Claude Code (disclosed) — scaffolding and copy.
- We also trained an EEG depression model with scikit-learn on public data — shown only as a cautionary exhibit, NOT used in the product.
No paid tool is required: a built-in demo runs the full app for free.
```

## Data Disclosure (≤800 chars)
```
- Public resource directory (curated by us; official public services), each with real contact, source link, and trust metadata (who runs it / free / confidential / hours): 988, Crisis Text Line, SAMHSA Helpline, FindTreatment.gov, 211, NAMI, The Trevor Project, Teen Line, The Jed Foundation, Trans Lifeline.
- Example documents: SYNTHETIC — we wrote realistic mock letters (ER discharge, school referral, insurance denial). No real personal data.
- The benched EEG model was trained on the public Mumtaz 2016 EEG dataset (figshare, CC BY 4.0); it is NOT used in the product.
- Nothing is stored; pasted text is sent to the AI provider only to generate the explanation.
```

## Demo Materials
- **Pitch video (3–5 min):** _[paste link]_ — script in `DEMO_SCRIPT.md`.
- **Working demo / walkthrough:** GitHub repo (README + run instructions) and/or a deployed Streamlit link.
