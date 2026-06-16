# PLAN.md — Build Plan (Phase 1 gate)

**Project:** Personalized Depression Treatment Decision-Support (research prototype)
**Date:** 2026-06-16
**Inputs:** `RESEARCH.md` (Phase 0). **Context:** ~1 week, hackathon/competition judging, Streamlit, "find the data" delegated.
**⚠️ Research prototype — not for clinical use.**

---

## 1. Scope decision (what the demo will and will not do)

**WILL (real):**
- **Layer A** — a genuinely real pipeline on open EEG data: MNE preprocessing → literature-grounded features → regularized classifier for **MDD vs. healthy control** (the only task the data supports), validated **subject-wise**, with honest metrics + a chance baseline + permutation test + probability calibration.
- **Signature feature** — a **leakage demonstration**: the *same* model under random k-fold (leaky) vs. leave-one-subject-out (correct), showing the accuracy collapse on the Mumtaz dataset (the "98%" claim originated here).
- **Layer B** — a Streamlit clinician dashboard that shows the *envisioned* workflow on **synthetic patient cases**: EEG feature summary (fed by Layer A for real cases) + synthetic hormonal markers → an **illustrative** subtype/treatment-response leaning, with prominent uncertainty, clinician-readable explanations, and human-in-the-loop Confirm/Override/Request-review controls.

**WILL NOT (honesty ceiling):**
- ❌ Predict treatment response from real data (no open data supports it).
- ❌ Claim clinical validation, regulatory approval, or Non-Device CDS exemption (an EEG tool is a *device* under FDA CDS Criterion 1).
- ❌ Autonomously diagnose, prescribe, or dose.
- ❌ Present synthetic/illustrative panels as real model outputs.
- ❌ Claim validity for adolescents (all open data is adult → off-distribution).

**Framing principle:** the honesty of the A/B separation *is* the deliverable. A correctly-validated modest result beats a suspiciously perfect one.

---

## 2. Dataset decision

- **Primary:** **Mumtaz 2016** (figshare DOI 10.6084/m9.figshare.4244171, CC BY 4.0) — 19-ch, 256 Hz, resting EC/EO, 34 MDD / 30 HC adults. **Diagnosed MDD vs HC + direct download.**
- **Fallback / robustness:** **OpenNeuro ds003478** (CC0) — 64-ch, 122 young adults, BDI-defined (subclinical). Used as a second cohort *if time allows*; clearly labeled a symptom (not diagnosis) classifier.
- **Not used:** MODMA (gated, ~days approval — access risk for a 1-week build); EMBARC/CAN-BIND (hard-gated; the only real treatment-response data).
- **Hormonal:** **synthetic, calibrated** to published reference distributions (CIRCORT / MIDUS); MIDUS named as the real-data upgrade path.
- **Data handling:** raw EEG is **not committed** (license + size). `data/` holds a `download.md` + a fetch script; `.gitignore` excludes raw files. Cache derived per-subject feature tables (small, redistributable as our own derived artifact) so the app runs without re-downloading.

---

## 3. Architecture

```
USAII_Project/
├── RESEARCH.md, PLAN.md, README.md
├── requirements.txt            # pinned
├── data/
│   ├── download.md             # how to obtain Mumtaz (+ WAF/browser note)
│   ├── fetch_mumtaz.py         # scripted download w/ manual fallback
│   ├── raw/                    # .gitignored
│   └── features/               # cached per-subject feature tables (committed)
├── ml/                         # LAYER A (real)
│   ├── preprocess.py           # MNE pipeline
│   ├── features.py             # bandpower, FAA (unit-tested sign), coherence
│   ├── validate.py             # subject-wise CV, metrics, permutation, calibration
│   ├── leakage_demo.py         # k-fold vs LOSO accuracy-drop
│   ├── train.py                # end-to-end; saves model + metrics + figures
│   └── artifacts/              # model.pkl, metrics.json, figures/
├── app/                        # LAYER B (illustrative)
│   ├── Home.py                 # Streamlit entry + persistent banner
│   ├── pages/                  # patient input, profile, explainability, model card
│   ├── synthetic.py            # synthetic patient + cortisol generator (calibrated)
│   ├── cases/                  # curated demo cases (incl. real Layer-A-fed cases)
│   └── components/             # uncertainty, contribution chart, HITL controls
├── tests/                      # FAA sign, no-leakage, synthetic ranges
└── notebooks/ (optional)       # Layer A walkthrough
```

**Data flow:** Layer A `train.py` → `artifacts/` (model, metrics, calibrated probabilities, feature contributions). Layer B loads those for **real** cases and the `synthetic.py` generator for everything the real data can't support (hormones, treatment/subtype leanings). A clear in-UI tag marks each value **Real (Layer A)** vs **Illustrative (synthetic)**.

---

## 4. Tech stack + rationale

- **Python 3.11 or 3.12 in a venv** — **not** the system 3.14 (MNE/scikit-learn wheel availability risk on a brand-new Python). Pin in `requirements.txt`.
- **Layer A:** `mne`, `scikit-learn`, `numpy`, `scipy`, `pandas`; `mne-connectivity` (optional); `joblib`. Plots via `matplotlib`.
- **Layer B:** **Streamlit** (chosen: fastest path; ML + UI share one Python runtime; ideal for a 1-week honest demo). `plotly`/`altair` for interactive charts; `shap` *optional* (fallback: linear coef × standardized value).
- **Quality:** `pytest` for the three guard tests; `ruff` (lint) if time.

---

## 5. Validation method & honest metrics

- **CV:** `LeaveOneGroupOut` (n≈64) or `GroupKFold(5)`, grouped by subject ID. **All transforms inside a `Pipeline`** — scaler/selector refit per fold only. No oversampling across the subject boundary.
- **Report (always with a baseline):** ROC-AUC (headline), sensitivity, specificity, confusion matrix, **per-fold spread / CI**, `DummyClassifier(most_frequent)` floor, `permutation_test_score(..., groups=…, n_permutations=1000)` p-value.
- **Calibration:** `CalibratedClassifierCV(method="sigmoid")`; show a reliability curve.
- **Leakage demo:** identical pipeline, random k-fold vs LOSO; report both numbers and the delta verbatim in the UI/README.
- **Honesty rule:** if the model lands near chance, **report it as-is.** A modest, correctly-validated number + the leakage demo is the credible story.

---

## 6. Responsible-AI features → UI implementation

| Requirement | Implementation in Streamlit |
|---|---|
| Persistent "not for clinical use" | Pinned `st.warning` banner on every page + footer on any export |
| Uncertainty, not a verdict | Calibrated probability **+ confidence band**; reliability curve on demand |
| Low-confidence handling | Explicit "**Insufficient evidence / low confidence**" state in a threshold band or wide-CI/OOD input |
| Flag for human review | "**Needs clinician review**" badge + review queue for flagged cases |
| Explainability | Horizontal **feature-contribution** bar chart, plain-language labels + direction; raw inputs & data-quality flags shown alongside |
| Human-in-the-loop | "**Suggestion**" wording only; required **Confirm / Override / Request review** action before anything is logged |
| Model card / transparency | Dataset size, subject-wise metrics + CIs, permutation p, validation method, caveats (FAA not validated; adolescent off-distribution; illustrative panels) |

---

## 7. Milestones & time budget (~1 week)

| Day | Deliverable |
|---|---|
| **1** | Env (venv 3.11/3.12, pinned deps); **verify Mumtaz download** (the #1 access risk); repo scaffold; quick EDA. |
| **2** | Layer A: MNE preprocessing + feature extraction; FAA sign unit test; cache feature tables. |
| **3** | Layer A: subject-wise CV, metrics, permutation/Dummy baselines, calibration, **leakage demo**; save artifacts + figures. |
| **4** | Layer B: Streamlit skeleton + banner; synthetic patient/cortisol generator; patient-input panel (real Layer-A feed for real cases). |
| **5** | Layer B: candidate-profile (illustrative) + uncertainty + explainability + HITL controls + model card. |
| **6** | Integration; curated **demo mode** (real + synthetic cases); polish; tests green. |
| **7** | README (honest "real vs illustrative"); reproducibility pass; demo script/video; buffer. |

**Cut list (drop first → last):** alpha-coherence feature → SHAP (use linear coef×value) → ds003478 cross-dataset check → PDF export → cortisol-calibration sophistication. **Never cut:** real Layer A on Mumtaz w/ subject-wise CV; the leakage demo; Layer B's uncertainty + HITL + banner; README honesty.

---

## 8. Risks & handling

| Risk | Likelihood | Handling |
|---|---|---|
| **Mumtaz download blocked by WAF** (automated fetch 202/403) | Medium | Day-1 verification; `fetch_mumtaz.py` + documented manual browser download; ds003478 (CC0, frictionless) as immediate fallback. |
| **Python 3.14 dependency wheels missing** | Medium | Pin a 3.11/3.12 venv up front; don't rely on system Python. |
| **Model near chance** | Medium | Report honestly — it's a *valid* result; the leakage demo + responsible framing carry the narrative regardless. |
| **Accidental data leakage inflating metrics** | Medium | Pipeline-wrapped transforms; subject-grouped splits; the demo itself is a guard; a `tests/` check asserts no subject crosses folds. |
| **Time overrun** | Medium | Cut list above; Layer A + core Layer B are the MVP, everything else is additive. |
| **Overclaiming creep in UI copy** | Low | "Suggestion"/"illustrative" lint pass before demo; banner + model card mandatory. |

---

## 9. Definition of done (from the brief)

- [ ] `RESEARCH.md` ✅ (Phase 0)
- [ ] `PLAN.md` ✅ (this doc)
- [ ] Layer A runs on real open data, subject-wise validated, honest metrics + leakage demo
- [ ] Layer B dashboard: uncertainty + explainability + HITL on synthetic cases
- [ ] All responsible-AI requirements visibly implemented
- [ ] `README.md` states real vs. illustrative + setup/run steps
- [ ] Honest about limitations throughout
