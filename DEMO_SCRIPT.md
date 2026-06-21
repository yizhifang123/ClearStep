# 🎬 MindBridge — 3–5 Minute Pitch Video Script

**Target:** ~3:40. **Covers the 4 required beats:** (1) problem & user, (2) how the AI
works, (3) live walkthrough, (4) responsible-AI choice.
**Judge's lens:** lead with WHO is helped in the first 30 seconds.

**Setup:** `streamlit run app.py` (demo mode — no API key needed; the model signal is
live, the written text is cached so every take is identical). Have **"Patient 1 —
elevated EEG signal"** selected.

---

### 🎯 Beat 1 — The problem & the user (0:00–0:40)

**On screen:** you, or a title card: "MindBridge — when help is written for doctors, not families."

> "Imagine you're a parent. Your fifteen-year-old just had a mental-health evaluation,
> and you're handed the results: brainwave power readings, cortisol levels, a line that
> says *'frontal alpha asymmetry: negative.'* You have no idea what any of it means, or
> what you're supposed to do next. The help is right there in your hands — but it's
> written for specialists, not for you. So you leave confused, and the next step doesn't
> happen. That's who we built MindBridge for: the stressed family on the other side of a
> clinical workup."

---

### 🤖 Beat 2 — How the AI works (0:40–1:30)

**On screen:** the README architecture diagram, or narrate over the app.

> "MindBridge takes one workup and serves two people at once. First, a *real* machine-
> learning model — one we trained on public EEG data — reads the brainwave features and
> returns a signal: how much this looks like the pattern seen in depression, with its
> uncertainty. Then a retrieval step — this is RAG — pulls the matching *clinical
> guidelines* and *public support resources* from curated lists, so nothing is invented.
> Finally, Claude turns the signal plus that evidence into two outputs: for the clinician,
> a summary where every point cites a guideline; for the family, the same thing in plain,
> sixth-grade language with clear next steps. Classification, retrieval, and generation —
> three AI capabilities, one workup."

---

### 🖥️ Beat 3 — Live walkthrough (1:30–2:50)

**On screen:** the running app.

> "Here it is. I'll pick a synthetic patient — all our data is synthetic, no real
> patients. You can see the inputs: the brainwave bands, the cortisol panel. I click
> *Run AI analysis.*
>
> The model gives its signal — here, an elevated probability, but notice it shows the
> *confidence* and the validation range right away: this model scores well on one small
> dataset, which is *not* the same as being a clinical test. It even flags features that
> fall outside what it was trained on.
>
> Now the two views. The **clinician view** lists evidence to consider — and every line
> cites a real guideline, like GLAD-PC, plus safety checks. The **family view** —" *(click
> the Family tab)* "— is the part that changes someone's day: *'A computer tool noticed a
> pattern that can go along with depression. It is not sure, and it is not a diagnosis.
> What matters most is a real check-in with your clinician.'* Then what matters, a
> checklist with timeframes, and trusted resources — with 988 right at the top."

---

### 🛡️ Beat 4 — The responsible-AI choice (2:50–3:35)

**On screen:** scroll to the human-in-the-loop buttons; if you can, switch to **Patient 3
(low signal)** to show the "low ≠ all clear" line.

> "And here's the design choice we're proudest of. This is medical-adjacent, so we drew a
> hard line: MindBridge **never diagnoses and never prescribes.** It gives a clue and the
> evidence; the clinician makes every decision — and we log that human step.
>
> Our biggest risk is a family misreading the signal — especially reading a *low* result
> as 'all clear.' So the tool says it outright:" *(show Patient 3)* "*'A low result does
> not mean everything is fine.'* The uncertainty is always on screen, the AI never gives
> medical advice, and 988 is always one tap away.
>
> MindBridge takes help that was locked behind specialist language and hands it to the
> family in words they can use — confusion, to clarity, to action — without ever
> pretending to be the doctor. Thanks for watching."

---

## Tips
- **~580 words ≈ 3:40** at a calm pace. The model signal is live every take; text is cached.
- Show the **988 card** and the **uncertainty/validation caption** on camera — they're your
  strongest Responsible-AI evidence (and judges score it).
- The **"low signal ≠ all clear"** moment (Patient 3) is your single best responsible-AI beat. Don't skip it.
- End on **"confusion → clarity → action"** — straight from the challenge brief's Impact section.
- With an API key, analyze one *custom* synthetic patient live to prove it's not canned.
