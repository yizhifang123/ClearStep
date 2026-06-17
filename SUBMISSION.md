# Personalized Depression Treatment Decision-Support — Submission

> ⚠️ **Research prototype — not for clinical use.** No diagnosis, no prescription, no validated medical claims.

**Tagline:** An honest decision-*support* prototype that analyzes EEG to help a psychiatrist personalize depression care — built to openly state where the science isn't there yet.

## Inspiration
Depression treatment is largely trial-and-error: patients cycle through medications, often enduring several failed trials before relief. A teammate watched a friend get the *same* treatment over and over with no personalization. We asked: could biological signals (EEG, hormones) help find the right treatment faster? The honest answer the field gives is "not yet" — so we built something that respects that limit instead of faking it.

## What it does
Two clearly-separated layers:
- **Layer A — real ML.** Loads the open **Mumtaz 2016** EEG dataset, extracts literature-grounded features, and classifies **MDD vs. healthy control** with leakage-free **subject-wise** validation. It also *demonstrates* the field's most common methodological error (data leakage).
- **Layer B — illustrative dashboard.** A Streamlit clinician view (on **synthetic** patients) showing the envisioned workflow: an EEG-based indication, an illustrative cortisol/subtype panel, calibrated uncertainty, plain-language explanations, automatic low-confidence flagging, and **human-in-the-loop** Confirm / Override / Request-review controls.

## How we built it
`Python · MNE-Python · scikit-learn · Streamlit`. EEG → notch/band-pass/average-reference (MNE) → 2-second epochs → 16 features (relative band power across regions + frontal alpha asymmetry) → averaged per subject → L2 logistic regression → **leave-one-subject-out** evaluation + permutation test + probability calibration. The dashboard reads the trained model and renders the workflow.

## The honest result (and why we don't oversell it)
On 58 subjects: **ROC-AUC 0.95, accuracy 93%** (sens/spec ~93%, permutation p=0.001). That's high — and we treat it skeptically rather than as a win:
- It's *correctly* validated (subject-wise, no leakage), so it isn't the usual overfit artifact.
- But it's driven mostly by **relative beta power**, which can be muscle/EMG or acquisition confound as easily as depression — and Mumtaz is a small, single-site, **adult** dataset known to over-separate.
- **Frontal alpha asymmetry contributes ~nothing** — corroborating the research that it's a near-null biomarker.
- The effect is **robust across resting conditions** (eyes-open AUC 0.92 vs eyes-closed 0.95) — a good sign, though a stable confound would also be condition-robust.

So our credibility is *rigorous validation + transparent caveats*, not a reassuring number.

## What's real vs. illustrative
**Real:** the EEG→MDD-vs-HC model, its subject-wise metrics, the leakage demo, the beta-driver finding. **Illustrative/synthetic:** all hormonal inputs, subtype leanings, treatment framing, and the adolescent cases.

## Challenges
- **No open data links biology to treatment response** (EMBARC/CAN-BIND are hard-gated) → we honestly scoped to diagnosis, not treatment prediction.
- **Data leakage** inflates most EEG-depression papers → we used subject-wise CV and built a demo of the failure mode.
- **A suspiciously high AUC** → we dug into the drivers and reported the confound instead of celebrating.

## Accomplishments
A working, end-to-end, *honest* prototype with a real validated ML slice, a responsible clinician UI, a domain memo with sources, and a test suite — that an expert in precision psychiatry would find credible *because* of its restraint.

## What we learned
Correct validation ≠ clinical utility; the hard part of "biomarker → treatment" is unsolved; and in a field full of overclaiming demos, honesty is the most impressive thing you can ship.

## What's next
External validation / cross-dataset transfer (expected to drop — and that's the point), adolescent-appropriate data, and — only with real treatment-outcome data — a genuine treatment-response model.

## Built with
Python, MNE-Python, scikit-learn, NumPy/SciPy/pandas, Streamlit, Matplotlib. Data: Mumtaz et al. 2016 (CC BY 4.0).

## Try it
See `README.md`. `RESEARCH.md` has the sourced evidence; `PLAN.md` the scope; `DEMO.md` the walkthrough; `SLIDES.md` the deck; `VIDEO_SCRIPT.md` the recording script.
