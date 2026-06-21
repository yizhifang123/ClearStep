# MindBridge — Devpost Submission Copy

Paste each block into the matching Devpost field. Char-limited fields are kept under limit.

---

## Qualifier Approval Code
```
HS26-86A98BB5
```
*(Double-check against your original email before submitting.)*

## Track & Challenge
- **Track:** High School (Grades 9–12)
- **Challenge:** Challenge 1 — *Help is Hard to Find* · Direction A (Crisis-to-Action / translate confusing info into plain language + next steps)

## Project Title
```
MindBridge
```

## Tagline (≤80 chars)
```
A teen's confusing depression workup, turned into steps a parent can act on
```

## Elevator pitch
```
Sam, 16, has been irritable and skipping school for weeks. After a depression screening, his parent is handed brainwave readings, cortisol numbers, and terms like "frontal alpha asymmetry" — and no idea what to do next. Here's the catch: there is no single test that says "depression." The signal is spread thin across dozens of noisy brain and hormone measures, and the weight of each one has to be LEARNED from real cases, not guessed — something no paper checklist or single-lab cutoff can do, and exactly what a trained model is for. MindBridge scores Sam's workup into a risk signal with its uncertainty, retrieves the matching clinical guidelines, and writes two grounded views of the same case: guideline-cited evidence for the clinician, and plain-language next steps for the parent. And it says plainly that a LOW score does not mean "all clear." The AI explains and flags its doubt; the clinician decides.
```

---

## Project Description (the "About the Project" field)

**Who it's for & why it matters**
Meet **Sam, 16** — six weeks of irritability, skipping school, pulling away from friends.
After a depression screening, **Sam's parent is handed a page of brainwave power bands,
cortisol levels, and terms like *"frontal alpha asymmetry."*** The information that could
help is right there in their hands — but it's written for specialists. The parent leaves
confused, and the single most important thing, *the next step*, often never happens.
**MindBridge** takes that one workup and produces **two** plain views of the same
analysis — a guideline-cited evidence summary for the clinician, and a plain-language
explainer for the family — moving a scared parent from *confusion → clarity → action.*

**Why this needs AI — not a checklist, not a single lab test**
This is the heart of MindBridge, and it's the question we most want to answer. Depression
has **no validated single biomarker** — no one number you can threshold. The signal is
*distributed and faint*: it lives in the combination of dozens of EEG band-power measures,
the shape of the cortisol response, and behavioral context — each weak and noisy on its
own. The natural objection is "so use a weighted checklist." But a checklist's weights are
**guessed by a committee**, and its output is a **bucket** (low/medium/high). The reason
this needs machine learning is narrower and sharper: *the relative weight of each noisy,
correlated feature has to be learned from real labeled cases, and the result has to be a
probability carrying its own uncertainty — not a yes/no.* No hand-set formula or
single-biomarker cutoff can do that. We deliberately use a simple, **auditable
logistic-regression** model rather than a black-box net — the "AI" here is in the *learned
fit to data*, not in complexity. Then a second, different AI problem
appears: turning *this* patient's specific numbers into language a frightened parent can
act on. The space of possible cases is combinatorial, so a static pamphlet or template
can't cover it — only **generative NLP, with the clinical claims supplied by retrieval**,
can explain *this* result for *this* family.

**How the AI works (input → AI → output)**
Three AI capabilities, one flow. (1) **Classification:** a logistic-regression classifier —
weights *learned* from labeled EEG data, validated leave-one-subject-out on the public
Mumtaz dataset (AUC ≈ 0.95, matching published benchmarks on this small dataset; **not a
clinical result**) — scores the brainwave pattern into a **probability with an
uncertainty flag**. The app then places the cortisol panel and behavioral note *alongside*
that score as context the clinician weighs — they're shown, not folded into the model by
guessed weights. (2) **Retrieval (RAG):** TF-IDF matches the case to
a curated, *cited* clinical-guideline corpus and a public support-resource directory; when
a clinician pastes a *synthetic* physician note, a second document-level retrieval grounds
the output in that note and translates its jargon for the family. (3) **Generative AI:** an
LLM, **restricted to the retrieved guideline spans**, writes the prose — it never
originates a medical claim. Every clinician point cites a guideline; the family text is
written at a 6th-grade reading level with a *patient-specific* next-step checklist and
linked resources.

**The concrete before → after**
*Before:* Sam's parent holds a page of numbers, understands none of it, and the follow-up
slips. *After:* one screen says — in plain words — *this is one clue, not a diagnosis;
here's what "frontal alpha asymmetry" means; book the follow-up this week, write down the
sleep and mood changes to bring, and save 988 — and a low score does **not** mean
all-clear.* The action that used to evaporate now has a checklist behind it.

**How we built it**
A Streamlit app. The classifier runs locally (always live, no API key), so the core AI
capability is always demonstrable; the written output uses a generative model, with a
cached demo mode so the built-in patients work with zero setup. We reused the trained EEG
model and synthetic-patient generator from our companion research project.

**Challenges**
The honest tension: a model that scores well on one small dataset is *not* a clinical test.
We designed the whole product around that — the AI is a *clue*, never a verdict — including
the counter-intuitive but vital rule that a **low signal does not rule out depression**, and
an explicit limitation: the model is trained on *adult* data and is **not** validated for
adolescents — so we demo a teen case deliberately, to keep that validation gap visible
rather than to claim adolescent accuracy.

**Accomplishments we're proud of**
Real ML rigor (leave-one-subject-out validation, a probability with its uncertainty, out-of-distribution
flags) paired with genuine accessibility — and guardrails that hold *even if the model is
wrong* (988 always shown, uncertainty always visible, clinician always decides).

**What we learned**
The most valuable AI here isn't the most confident answer — it's the *clearest* one,
delivered with the human firmly in control, and honest about what it doesn't know.

**What's next**
Real PHQ-A intake, local resources by ZIP (211 API), multi-language family output, and a
clinician export that prints the plain-language explainer as a take-home sheet.

**Built with:** Python · Streamlit · scikit-learn (trained classifier + RAG retrieval) ·
generative LLM API · Claude Code (AI coding assistance, disclosed)

---

## AI Architecture Explanation (≤600 chars)
```
Inputs: synthetic EEG features, cortisol, behavioral note. AI: classification + retrieval (RAG) + generative. Processing: a logistic-regression classifier — weights LEARNED from labeled EEG data, not hand-set — scores the brainwave pattern into a probability + uncertainty; cortisol and behavior are shown alongside as context, not fused into the score; TF-IDF retrieves CITED guidelines + resources; an LLM restricted to those spans writes the text. Outputs: a clinician summary (every point cites a guideline) and a 6th-grade family explainer with patient-specific steps and a 988 crisis card.
```

## Human-in-the-Loop Design (≤500 chars)
```
MindBridge does NOT diagnose and does NOT choose treatment. It surfaces an uncertainty-flagged signal plus cited guideline evidence; the clinician makes every clinical decision. The family view makes no recommendation itself — it explains what the workup means and routes results back to the clinician and 988. Why: no EEG or hormonal marker is validated to diagnose or pick treatment for an individual, and only a clinician can weigh the whole person and context the model never sees.
```

## Responsible AI Guardrail (≤500 chars)
```
Risk: a family reads the signal as a diagnosis — especially treating a LOW score as "all clear" and skipping needed care. Mitigation: the tool states plainly that a low score does NOT rule out depression, shows its uncertainty on every result, gives no diagnosis or medication advice, and always surfaces 988 plus a clinician handoff. We also disclose its limits: trained on adult data, it is not validated for adolescents. All patient data is synthetic — no real personal data is exposed.
```

## Tools Used (≤800 chars)
```
- scikit-learn (free, open-source) — the EEG classifier (logistic regression, which WE trained) AND the TF-IDF retrieval for RAG.
- Generative LLM API (Google Gemini, model gemini-2.5-flash) — FREE tier. Generates the family explainer and the guideline-grounded clinician summary, restricted to retrieved spans. The app is provider-agnostic: one interface also supports Anthropic Claude and OpenAI.
- Streamlit, joblib, numpy, python-dotenv (all free, open-source) — web app, model loading, math, config.
- AI coding assistance: Claude Code (disclosed) — scaffolding and copy.

No paid tool is required: the model runs locally and the LLM free tier covers the text, so the whole app runs free.
```

## Data Disclosure (≤800 chars)
```
- Model training (real, public): Mumtaz 2016 "MDD Patients and Healthy Controls EEG" dataset (figshare, CC BY 4.0; a small adult cohort). Used only to train the EEG classifier; raw data not redistributed. Important limitation: it is ADULT data, so adolescent performance is unvalidated.
- App patients (synthetic): every patient in the app is synthetic — EEG features generated via Dirichlet draws; cortisol calibrated to published adolescent reference ranges. No real patient data.
- Public corpora (curated by us): clinical-guideline snippets cited to GLAD-PC, NICE, APA, USPSTF, AACAP; public support resources (988, Crisis Text Line, 211, SAMHSA, NAMI, Trevor Project) with source links.
- No sensitive or personal data is collected or stored.
```

## Demo Materials
- **Pitch video (3–5 min):** _[paste link]_ — script in `DEMO_SCRIPT.md`.
- **Working demo / walkthrough:** GitHub repo (this README + run instructions) and/or a deployed Streamlit link.
