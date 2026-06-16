# Personalized Depression Treatment Decision-Support — Research Prototype

> ⚠️ **Research prototype — NOT for clinical use.** It does not diagnose, prescribe, or make validated medical claims. It is a decision-*support* concept demo for clinicians, on a vulnerable population — read `RESEARCH.md` for the honest scientific limits.

## What this is — real vs. illustrative

Two clearly-separated layers; the honesty of the separation is the point.

- **Layer A — real ML slice.** A genuine EEG → features → classifier pipeline trained on the open **Mumtaz 2016** dataset to distinguish **MDD vs. healthy control** — the only task open data supports, **not** treatment response. Validated with leakage-free **subject-wise** cross-validation, with honest metrics and a built-in demonstration of how the field's headline "98%" accuracies inflate under improper validation.
- **Layer B — illustrative clinician dashboard.** A Streamlit decision-support UI on **synthetic patient cases**. Treatment/subtype leanings and hormonal (cortisol) inputs are **simulated and explicitly labeled illustrative** — they show the *envisioned* workflow with visible uncertainty and clinician control, not validated predictions.

See `RESEARCH.md` (domain evidence + sources) and `PLAN.md` (scope, architecture, validation, risks).

## Status
🚧 Phase 2 (build): Layer A pipeline + Layer B dashboard implemented and tested (test suite green). Awaiting the EEG data download to generate the real metrics/figures — the app runs now, and the synthetic workflow is fully functional without it.

## Project structure
See `PLAN.md` §3.

## Setup & run

Requires **Python 3.11** (the repo pins a 3.11 virtualenv; the system 3.14 lacks some wheels).

```bash
# 1. Environment
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 2. Data — browser download (see data/download.md); place .edf files in:
#    data/raw/mumtaz/

# 3. Layer A — train + evaluate (writes metrics, figures, calibrated model)
.venv/bin/python -m ml.train --data-root data/raw/mumtaz --condition EC

# 4. Layer B — clinician dashboard (run from the project root)
.venv/bin/streamlit run app/Home.py

# Tests
.venv/bin/python -m pytest -q
```

The dashboard runs **without** step 3: Layer A panels show a "run training" hint, and the illustrative (synthetic) workflow works standalone.

## Responsible-AI commitments
Human-in-the-loop (the clinician decides), visible & calibrated uncertainty, explainability, automatic low-confidence flagging, synthetic-or-licensed data only, and a persistent "not for clinical use" notice. See `PLAN.md` §6.
