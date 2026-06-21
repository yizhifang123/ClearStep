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
Turns a confusing mental-health workup into plain language a family can act on
```

## Elevator pitch
```
A parent whose teen just had a mental-health workup is handed brainwave readings and terms like "frontal alpha asymmetry" — and no idea what to do. MindBridge turns that specialist workup into plain language, what matters most, and clear next steps for the family, while giving the clinician guideline-cited evidence. The AI explains and supports; the clinician decides.
```

---

## Project Description (the "About the Project" field)

**What it does & why it matters**
Help is often *there* — but locked behind language built for specialists. Meet a
**parent whose 15-year-old just had a mental-health workup**: the results list brainwave
power bands, cortisol levels, and terms like *"frontal alpha asymmetry."* The parent
leaves confused, and the next step never happens. **MindBridge** bridges that gap: it
takes one depression workup and produces **two** views of the same analysis — an
evidence summary for the clinician, and a plain-language explainer for the family —
moving a scared family from *confusion → clarity → action.*

**How the AI works**
Three AI capabilities in one flow. (1) **Classification:** a *real* EEG model we trained
(scikit-learn logistic regression on the public Mumtaz dataset, leave-one-subject-out
AUC ≈ 0.95, p < 0.001) returns a calibrated, uncertainty-flagged signal. (2)
**Retrieval (RAG):** TF-IDF matches the case to a curated, *cited* clinical-guideline
corpus and a public support-resource directory — and, when a clinician pastes a
*synthetic* physician note, a second document-level RAG grounds the output in that note's
specifics, translating its jargon for the family. (3) **Generative AI:** a large language
model (Google Gemini) turns the signal + retrieved evidence into the two grounded
outputs — every clinician point cites a guideline; the family text is written at a
6th-grade level with a next-step checklist.

**Why AI, not a pamphlet:** a generic handout can't read *this* workup, judge *its*
uncertainty, and translate *it* for *this* family. That per-case work is what AI does.

**How we built it**
A Streamlit app. The model runs locally (always live, no API key), so the AI capability
is always demonstrable; the written output uses Google Gemini (free tier), with a cached
demo mode so the three built-in patients work with zero setup. We reused the trained EEG model and
synthetic-patient generator from our companion research project.

**Challenges**
The honest tension: a model that scores 95% on one dataset is *not* a clinical test. We
designed the whole product around that — the AI is a *clue*, never a verdict — including
the counter-intuitive but vital rule that a **low signal does not rule out depression.**

**Accomplishments we're proud of**
Real ML rigor (LOSO validation, calibration, out-of-distribution flags, permutation
p-value) paired with genuine accessibility — and guardrails that hold even if the model
is wrong (988 always shown, uncertainty always visible, clinician always decides).

**What we learned**
The most valuable AI here isn't the most confident answer — it's the *clearest* one,
delivered with the human firmly in control.

**What's next**
Real PHQ-A intake, local resources by ZIP (211 API), multi-language family output, and a
clinician export that prints the plain-language explainer as a take-home sheet.

**Built with:** Python · Streamlit · scikit-learn (trained model + RAG) · Anthropic
Claude API · Claude Code (AI coding assistance, disclosed)

---

## AI Architecture Explanation (≤600 chars)
```
Inputs: a synthetic patient's EEG brainwave features + cortisol panel + context. AI capabilities: classification (a real trained EEG model) + retrieval (RAG) + generative AI. Processing: the EEG model outputs a calibrated, uncertainty-flagged signal; TF-IDF retrieves matching CITED clinical guidelines and public support resources; Gemini turns the signal + retrieved evidence into two grounded outputs. Outputs: a clinician evidence summary (every point cites a guideline) AND a family plain-language explainer with a next-step checklist and linked resources — plus an always-on 988 crisis card.
```

## Human-in-the-Loop Design (≤500 chars)
```
MindBridge does NOT diagnose and does NOT choose treatment. It surfaces an uncertain EEG signal plus cited guideline evidence; the clinician makes every clinical decision, and the family view says so plainly. Why: no EEG or hormonal marker is validated to diagnose or select treatment for an individual — the signal is a group-level pattern with wide uncertainty, and only a clinician can weigh the whole person, history, and context that the model never sees.
```

## Responsible AI Guardrail (≤500 chars)
```
Risk: a family reads the AI's signal as a diagnosis — especially treating a LOW signal as "all clear" and skipping needed care. Mitigation: the tool states explicitly that a low signal does NOT rule out depression, shows the model's uncertainty and low-confidence band on every result, never gives a diagnosis or medication advice, and always surfaces the 988 crisis line plus a clinician handoff. All patient data is synthetic, so no real personal data is exposed.
```

## Tools Used (≤800 chars)
```
- scikit-learn (free, open-source) — the EEG classifier (logistic regression, which WE trained) AND the TF-IDF retrieval for RAG.
- Google Gemini API (model: gemini-2.5-flash) — FREE tier. Generates the family explainer and the guideline-grounded clinician summary. The app is provider-agnostic: one interface also supports Anthropic Claude and OpenAI.
- Streamlit, joblib, numpy, python-dotenv (all free, open-source) — web app, model loading, math, config.
- AI coding assistance: Claude Code (disclosed) — scaffolding and copy.

No paid tool is required: the model runs locally and Gemini's free tier covers the text, so the whole app runs free.
```

## Data Disclosure (≤800 chars)
```
- Model training (real, public): Mumtaz 2016 "MDD Patients and Healthy Controls EEG" dataset (figshare, CC BY 4.0; 58 adults). Used only to train the EEG classifier; raw data not redistributed.
- App patients (synthetic): every patient in the app is synthetic — EEG features generated via Dirichlet draws; cortisol calibrated to published adolescent reference ranges. No real patient data.
- Public corpora (curated by us): clinical-guideline snippets cited to GLAD-PC, NICE, APA, FDA, USPSTF, AACAP; and public support resources (988, Crisis Text Line, 211, SAMHSA, NAMI, Trevor Project) with source links.
- No sensitive or personal data is collected or stored.
```

## Demo Materials
- **Pitch video (3–5 min):** _[paste link]_ — script in `DEMO_SCRIPT.md`.
- **Working demo / walkthrough:** GitHub repo (this README + run instructions) and/or a deployed Streamlit link.
