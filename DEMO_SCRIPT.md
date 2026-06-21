# 🎬 ClearStep — 3–5 Minute Pitch Video Script

**Target length:** ~3:30 (safely inside the 3–5 min window).
**Covers the 4 required beats:** (1) problem & user, (2) how the AI works, (3) live
walkthrough, (4) responsible-AI choice.
**Judge's lens:** lead with WHO is helped and HOW in the first 30 seconds. Don't bury it.

**Setup before recording:**
- Run the app: `streamlit run app.py` (demo mode is fine — no API key needed).
- Have the **ER discharge** example ready to load. The cached AI result is realistic and
  identical every take, so nothing breaks on camera.
- Record screen + voiceover. Speak calmly — the topic is stressful; your tone is the reassurance.

---

### 🎯 Beat 1 — The problem & the user (0:00–0:40)

**On screen:** Your face or a simple title slide: "ClearStep — when help is hard to find."

**Narration:**
> "Meet Jordan. Jordan is fifteen, and last night Jordan was in the emergency room after
> a mental-health crisis. This morning, Jordan's mom is holding the discharge paperwork —
> two pages that say things like *'establish outpatient behavioral health within seven
> days.'* She reads it twice and still doesn't know what she's actually supposed to do.
>
> The help Jordan needs exists. But it's written in clinical language, the resources are
> scattered, and it arrived at the worst possible moment to read carefully. That's the
> problem we're solving: not that help doesn't exist — that it's *hard to find and hard
> to understand.* We built ClearStep for that stressed parent and that scared teen."

---

### 🤖 Beat 2 — How the AI works (0:40–1:25)

**On screen:** The architecture diagram from the README (or just narrate over the app).

**Narration:**
> "Here's how ClearStep works. You paste the confusing document. First, a retrieval step —
> this is RAG — uses TF-IDF matching to pull the most relevant services from a curated
> directory of *real, public* resources: 988, Crisis Text Line, 211, SAMHSA. Crucially,
> we never search private or patient data — only public help.
>
> Then those resources, plus the document, go to Claude. Claude does four AI jobs at once:
> it *understands* the medical language, *classifies* how urgent the situation is,
> *summarizes* it at a sixth-grade reading level, and *generates* a step-by-step checklist —
> but it's allowed to reference *only* the resources we retrieved. No invented phone numbers,
> no made-up advice.
>
> And this is why it has to be AI, not a web search: a search engine can't read *this
> specific letter*, judge *its* urgency, and rewrite *it* for this family. That's the part
> only language AI can do."

---

### 🖥️ Beat 3 — Live walkthrough (1:25–2:45)

**On screen:** The running app. Do each action slowly enough to follow.

**Narration + actions:**
> "Let's see it. I'll load Jordan's ER discharge summary…" *(select the example)* "…and
> click *Explain this for me.*"
>
> *(Result appears.)* "Instantly — first thing, before anything else — the crisis line.
> Call or text 988. That's always here, no matter what.
>
> Then the urgency flag: *time-sensitive.* The AI explains why: there's a seven-day window
> to start care.
>
> Here's the document in plain language: *'Jordan was seen for an anxiety crisis. Jordan is
> safe to go home. Start regular mental-health care soon, keep taking the current medicines,
> and make the home safer.'* And notice —" *(open the expander)* "— the original document is
> always one click away. We never hide what we summarized.
>
> *'What matters most'* pulls out the things you can't miss — including locking up
> medications and firearms, which the ER specifically asked for.
>
> Then the part that actually helps: a checklist with *timeframes.* Call the access line
> within seven days. Secure medications today. And matched resources — each with a real link
> you can verify, like FindTreatment.gov to book that first appointment."

---

### 🛡️ Beat 4 — The responsible-AI choice (2:45–3:25)

**On screen:** Scroll to the "Talk to a real person" section and the flag button.

**Narration:**
> "Now the most important design choice. ClearStep is medical-adjacent, so we drew a hard
> line: it *explains*, it never *advises*, and it never makes the call that matters.
>
> One real risk: the AI could under-rate an urgent situation, or skip a critical
> instruction. So we put the guardrails in the *interface*, where they hold even if the
> model is wrong — the 988 line shows on every result regardless of the AI's rating, the
> original text is always visible, and every resource is source-linked.
>
> And the human stays in control: ClearStep does *not* decide whether this is a true
> emergency, and gives *no* medical advice. Those decisions route to a trained human — 988,
> or your provider — one tap away. Because misjudging a mental-health crisis can be fatal,
> and only a person can weigh what an AI can't see.
>
> ClearStep moves a scared family from confusion, to clarity, to action — without ever
> pretending to be the doctor. Thanks for watching."

---

## Quick tips
- **Total words ≈ 560**, which at a calm ~165 wpm is ~3:25. If you run short, slow down on Beat 3.
- If you have an API key, do **one live custom example** in Beat 3 to prove it's not canned —
  e.g., paste a short made-up insurance denial. Otherwise demo mode is completely fine.
- Show the always-on 988 card on camera at least twice — judges score Responsible AI, and
  it's your strongest, most visible safeguard.
- End on the phrase **"confusion → clarity → action"** — it's lifted straight from the
  challenge brief's own Impact section.
