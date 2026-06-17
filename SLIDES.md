---
marp: true
theme: default
paginate: true
---

<!-- Render: `marp SLIDES.md --pdf` (or VS Code Marp extension). -->

# Personalized Depression Treatment
## Decision-Support — Research Prototype

EEG + an honest clinician workflow

⚠️ *Research prototype — not for clinical use*

---

## The problem

- Depression treatment is **trial-and-error** — patients endure several failed medication trials before relief.
- Slow, expensive, sometimes harmful.
- Could biological signals (EEG, hormones) find the right treatment **faster**?

---

## The honest reality

- This is real, active science (**precision / computational psychiatry**)…
- …but **"biology → subtype → treatment" is the open research problem**, not an engineering task.
- No validated biomarker taxonomy. Even genetics-based "drug pickers" failed — including in an **adolescent RCT**.
- So we built honestly *within* those limits.

---

## Our approach: two layers

- **Layer A — real ML.** EEG → MDD-vs-healthy-control on open data, validated correctly.
- **Layer B — illustrative.** A clinician dashboard on *synthetic* cases showing the envisioned workflow.
- **The honesty of that separation is the product.**

---

## Layer A — the real pipeline

`EEG (MNE) → 16 features → logistic regression → leave-one-subject-out`

- Open **Mumtaz 2016** dataset, 58 subjects
- Relative band power + frontal alpha asymmetry
- **Subject-wise** validation (no leakage) + permutation test + calibration

**Result: ROC-AUC 0.95, accuracy 93%, p = 0.001**

---

## …but we don't oversell it

- Correctly validated — *not* the usual overfit artifact.
- **Driven by relative beta power** → could be EMG / acquisition confound, not depression.
- Single-site, **adult**, n=58, no external validation.
- Frontal alpha asymmetry contributes ~nothing — matching research that it's a **near-null biomarker**.

**Credibility = rigor + transparent caveats, not a big number.**

---

## The leakage demonstration

Same model, two validation schemes (epoch level):

- Leaky random k-fold: **88%**
- Honest subject-wise: **85%**

Leakage inflates accuracy — the mechanism behind the field's "98%" headlines. We validate subject-wise and show the gap.

---

## Layer B — clinician dashboard

- EEG indication with **calibrated uncertainty**, not a verdict
- **Low-confidence cases auto-flagged** for human review
- Plain-language **feature explanations**
- **Human-in-the-loop:** Confirm / Override / Request review
- Persistent **"not for clinical use"** banner

---

## Responsible AI

- Human decides; AI only suggests
- Honest, visible uncertainty
- Explainable
- Synthetic or properly-licensed data only
- An EEG tool is a **medical device** for real use → research-only by design

---

## Real vs. illustrative

- ✅ **Real:** EEG→MDD-vs-HC model, subject-wise metrics, leakage demo, beta-driver finding
- 🟠 **Illustrative:** hormonal inputs, subtype leanings, treatment framing, adolescent cases

---

## What's next

- Cross-dataset transfer (expected to drop — *that's the honest point*)
- Adolescent-appropriate data
- A real treatment-response model — only with real outcome data

**An honest prototype that states its limits beats an overclaiming demo.**
