# DEMO.md — How to present this prototype

Lead with honesty — that's what lands with people who know this field.

## The one-line pitch
> Depression treatment is trial-and-error. This is a *decision-support* prototype exploring how EEG (+ hormonal) signals could help a psychiatrist personalize treatment — built honestly, the clinician always in control, openly stating where the science isn't there yet.

## The two-layer story (say it explicitly)
- **Layer A is real.** A genuine EEG → features → classifier on the open Mumtaz dataset, **MDD vs. healthy control** — *not* treatment response (no open data supports that).
- **Layer B is illustrative.** A clinician dashboard on *synthetic* cases showing the envisioned workflow with loud uncertainty + human-in-the-loop, clearly labeled simulated.
- The honesty of that separation is the point.

## The honest result — and the move is NOT to oversell it
Subject-wise validation gives **AUC ≈ 0.95 / 93% accuracy** (permutation p < 0.01) on n=58. Say it, then immediately puncture it:
- "It's *correctly* validated — subject-wise, no leakage, permutation-tested — so it isn't the usual overfit artifact."
- "But it's driven mostly by **relative beta power**, which can be muscle/EMG or an acquisition confound as easily as depression — and Mumtaz is a small, single-site, **adult** dataset that's known to over-separate."
- "So: a high number, honestly reported, with the concrete reasons you still shouldn't trust it clinically. *That honesty is the deliverable.*"
- Bonus: **frontal alpha asymmetry barely contributes** — matching the research that FAA is a near-null biomarker. Our own result corroborates the honest literature.

## The leakage demonstration — a discipline check, not a gotcha
Same model, leaky k-fold vs. subject-wise: **≈88% vs. ≈85%** (epoch level). The gap is modest *because* we used low-dimensional features. Point to make: "leakage inflates in the expected direction; the famous 98%→chance collapses need high-capacity models — we avoided both the leakage *and* the over-capacity."

## Suggested 3-minute flow
1. **Home** — the real-vs-illustrative table. Set expectations.
2. **Model Card** — the AUC, then the "high accuracy ≠ clinical utility" caveat + the beta-driver table. This is the credibility moment.
3. **Real Case** — pick a held-out subject; show the calibrated probability, confidence state, and feature-contribution explanation; note low-confidence cases auto-flag for review.
4. **Illustrative Case** — a curated synthetic adolescent; EEG + synthetic cortisol → an *illustrative* subtype leaning; then state what the tool does **not** do (pick/dose a drug) and why (rACC theta is prognostic not prescriptive; pharmacogenomic pickers failed in an adolescent RCT).
5. **Close** on the banner + human-in-the-loop: "research prototype, not for clinical use; the AI suggests, the clinician decides."

## What's real vs. illustrative (if asked)
Real: the EEG→MDD-vs-HC model, its subject-wise metrics, the leakage demo, the beta-driver finding. Illustrative/synthetic: all hormonal inputs, subtype leanings, treatment framing, and adolescent cases. Sources in `RESEARCH.md`; scope in `PLAN.md`.
