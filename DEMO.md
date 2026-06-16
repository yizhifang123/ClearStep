# DEMO.md — How to present this prototype

A ~3-minute walkthrough that leads with honesty (the thing that lands with people who know this field).

## The one-line pitch
> Depression treatment is trial-and-error. This is a *decision-support* prototype that explores how EEG (+ hormonal) signals could help a psychiatrist personalize treatment — built honestly, with the clinician always in control, and openly stating where the science isn't there yet.

## The two-layer story (say this explicitly)
- **Layer A is real.** A genuine EEG→features→classifier pipeline on the open Mumtaz dataset, distinguishing **MDD vs. healthy control** — *not* treatment response, because no open data supports that.
- **Layer B is illustrative.** A clinician dashboard on *synthetic* cases showing the envisioned workflow (subtype/treatment leanings, cortisol) with loud uncertainty and human-in-the-loop — clearly labeled simulated.
- The honesty of that separation is the point.

## Suggested flow
1. **Home page** — show the real-vs-illustrative table. Set expectations up front.
2. **Model Card → the leakage demonstration** *(the headline)*. Same model, two validation schemes: leaky k-fold vs. honest subject-wise. Point at the accuracy gap and say: *"the famous '98% MDD detection' headlines come from the leaky split — reported on this very dataset. Here's the honest number."* This is the most memorable 30 seconds.
3. **Real Case page** — pick a held-out subject; show the calibrated probability, the confidence state, and the feature-contribution explanation. Note a low-confidence case is auto-flagged for review.
4. **Illustrative Case page** — generate a synthetic adolescent; show EEG + synthetic cortisol → an *illustrative* subtype leaning, then explicitly say what the tool does **not** do (pick or dose a drug), and why (rACC theta is prognostic not prescriptive; pharmacogenomic pickers failed in an adolescent RCT).
5. **Close on the banner + human-in-the-loop**: "research prototype, not for clinical use; the AI suggests, the clinician decides."

## Talking points that build credibility
- "There is no validated biomarker taxonomy that maps to treatment — the middle of this pipeline is the open research problem, so we didn't fake it."
- "We validated subject-wise to avoid the leakage that inflates most EEG-depression papers."
- "Frontal alpha asymmetry is in our feature set but it's *not* a validated biomarker (meta-analytic d ≈ 0) — it's one input, never a verdict."
- "An EEG tool processes a physiological signal, so for real clinical use it'd be a regulated *device*, not exempt decision-support — which is exactly why this is research-only."

## What's real vs. illustrative (if asked)
Real: the EEG→MDD-vs-HC model, its subject-wise metrics, and the leakage demo. Illustrative/synthetic: all hormonal inputs, subtype leanings, treatment framing, and the adolescent cases. See `RESEARCH.md` for sources and `PLAN.md` for scope.
