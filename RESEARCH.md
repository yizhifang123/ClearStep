# RESEARCH.md — Phase 0 Domain Memo

**Project:** Personalized Depression Treatment Decision-Support (research prototype)
**Date:** 2026-06-16
**Status:** Phase 0 (research) complete → gates Phase 1 (PLAN.md)
**⚠️ Research prototype — not for clinical use. No medical claims.**

This memo resolves the open questions in the project brief and audits its Section 3 assumptions against current primary/review sources. Citations are listed in §7. Where a claim could not be verified to primary-source precision, it is flagged.

---

## 0. TL;DR — the honest bottom line

1. **The idea is legitimate** (precision/computational psychiatry is a real, active field), **but its core step is unsolved.** There is **no validated biological taxonomy of depression that maps cleanly to treatment selection.** EEG, hormonal, and even genomic signals are group-level associations, not individual-level clinical tests.
2. **The strongest treatment-response EEG signal (rostral-ACC theta) is *prognostic, not prescriptive*** — it predicts who improves, including on *placebo*. Presenting any of these as a "which drug" selector would overclaim.
3. **We can build a genuinely real ML slice — but only for the task the data supports: MDD vs. healthy control**, on the openly-licensed **Mumtaz 2016** EEG dataset, with leakage-free **subject-wise** validation.
4. **The single most credible thing we can demonstrate** is the field's own central failure: re-run one model under leaky k-fold vs. correct leave-one-subject-out CV and show the accuracy collapse (literature analog: 99.8% → 53%). The famous "98% MDD-detection" headline was reported on *this very dataset*.
5. **Everything treatment- and hormone-related must be explicitly illustrative**, on synthetic cases, with loud uncertainty and a clinician in the loop.

---

## 1. Scientific landscape

### 1.1 EEG biomarkers — detection vs. treatment response

| Signal | Detection (MDD vs HC) | Treatment-response prediction | Verdict |
|---|---|---|---|
| **Frontal alpha asymmetry (FAA)** | Pooled **d ≈ −0.007** (van der Vinne 2017, n≈4,044); **d ≈ −0.03** (2025 meta, sig. only because N huge, I²=75%); multiverse: ~5% of 270 pipelines significant, most in the *wrong* direction (eLife 2021) | One well-powered positive (iSPOT-D, n=1,008) held **only in women, SSRIs only** | **Weak / effectively null for detection; preliminary & unreplicated for prediction** |
| **Rostral-ACC (rACC) theta** | — | Meta-analytic combined **d ≈ 1** across ~19 studies; **BUT** the decisive RCT (EMBARC, n=248) found it explains only **8.5% unique variance and is NOT moderated by treatment arm** → a **"nonspecific prognostic marker"** that predicts *placebo* response too | **Real but prognostic, not prescriptive** — the most honest signal to feature, with the placebo caveat |
| **ML "95–98% MDD vs HC"** | Real in print (Mumtaz 2018: 98%, SVM, n=63; Acharya 2018: 93–96%, CNN, n=30) — but the **largest study (Cai, n=213) dropped to ~79%** | — | **Largely artifact** of small-n + within-subject leakage; near-zero external validation |

**Why the headline accuracies inflate** (each mechanism sourced in §7): (a) tiny samples — >50% of psychiatric-prediction studies use n<50, with ±10% CV error bars at n=100; (b) **within-subject epoch leakage** — random k-fold puts the *same subject's* epochs in train and test, so the model learns the *person's* EEG fingerprint, not the disorder (Saeb 2017: error 5.6% → 13.0%, a >2.3× inflation, going record-wise → subject-wise; EEG-specific demo: 99.8% → 53% chance, Brookshire 2024); (c) feature selection on the whole dataset before CV; (d) publication bias + no external validation.

### 1.2 Depression "biotypes" / subtypes

- **Neuroimaging biotypes (Drysdale 2017, *Nature Medicine*)** — claimed 4 fMRI-connectivity biotypes predicting TMS response. **Failed to replicate:** Dinga 2019 (independent n=187) found the defining canonical correlations were **not significant under permutation** (collapsed to chance out-of-sample), cluster-validity indices **indistinguishable from a single Gaussian**, and "**no stable or statistically significant biotypes**." Original authors *conceded* standard CCA+feature-selection overfits; salvaged only a *continuous* brain–symptom axis, not discrete biotypes.
- **Melancholic vs. atypical** — DSM-5 *specifiers*, not separate disorders or a validated biological taxonomy. Historical "atypical → MAOIs" claim is not robustly validated in the SSRI/SNRI era; STAR\*D analysis found specifiers do **not** yield more homogeneous subgroups. Melancholia has the strongest biological correlates (HPA/DST, REM latency) but a poorly-bounded definition.

**Conclusion:** discrete, treatment-predictive depression subtypes are **not established.** A continuous biology–symptom relationship may exist; clean categories that pick a drug do not.

### 1.3 Hormonal / HPA-axis (cortisol)

- **Risk factor > state marker (in adolescents).** Elevated *morning* cortisol **prospectively predicts** later MDD (SMD 0.37) but does **not** distinguish current MDD cross-sectionally (Zajkowska 2022 meta-analysis). *Nuance:* **nocturnal** cortisol does track current illness — the dissociation is not universal across all cortisol indices.
- **Non-monotonic.** **Both elevated and blunted** cortisol are linked to MDD (hypercortisolism in severe/melancholic/psychotic; blunting in atypical/chronic-stress), moderated by sex and pubertal stage. "Depression = high cortisol" is an oversimplification.
- **Melancholic (hyper-) vs. atypical (hypo-/blunted) HPA** is supported **at the group level** (Stetler & Miller 2011: overall cortisol d≈0.60, melancholic > atypical; Lamers 2013/NESDA) **but is not a clinical test**: individual overlap is huge (even at d=0.60 only ~73% of patients exceed the control median), and data-driven subtyping (Beijers 2019/NESDA) found cortisol specifically **did not** separate melancholic vs. atypical. The **dexamethasone suppression test failed** as a diagnostic for exactly this reason (APA 1987): good specificity vs. *healthy* people is useless when it can't separate depression from the *other* disorders in the differential.

**Conclusion for Layer B:** cortisol/HPA framing is defensible *as illustration* of how a hormonal axis *could* inform a subtype leaning — but it can never subtype a real patient. Use **synthetic cortisol, calibrated to published reference distributions** (CIRCORT normative data / MIDUS diurnal slopes), explicitly labeled simulated.

### 1.4 Pharmacogenomics — the closest real-world analog, and a cautionary tale

- **Combinatorial panels (e.g., GeneSight) are contested.** The largest RCT (**GUIDED**, ~1,541 adults) **missed its primary endpoint** (HAM-D17 % change, p=**.069**); the marketed wins are **secondary/post-hoc subgroup** results. Best 2024 synthesis: pooled remission RR 1.41 (p=0.06, **ns**) and response RR 1.20 (p=0.11, **ns**); **all RCTs industry-funded, physicians unblinded, panels disagree on up to 67% of genotypes — GRADE: LOW.** FDA (2018/2019), CPIC, ISPG, and APA (2024) all decline to endorse routine use.
- **Adolescents specifically (our target population): NEGATIVE.** The only completed dedicated adolescent RCT (**Vande Voort 2022, *JAACAP*, n=176**) found combinatorial-PGx-guided treatment did **not** improve outcomes vs. treatment-as-usual. (A larger trial, ~452 youth, is ongoing.)
- **Honest distinction:** single-gene *pharmacokinetic* guidance (CYP2C19/CYP2D6) has cautious CPIC/ISPG support for *selected* patients; the contested part is the proprietary *combinatorial* algorithms. Don't conflate them.

**Lesson:** even the genetics-based version of this idea is contested — and *failed in teens.* This calibrates our claims hard: a credible prototype foregrounds uncertainty and clinician control, not predictive power.

---

## 2. Data availability — the binding constraint

**Question answered: is any open dataset labeled with treatment response? — No.** The only credible antidepressant-response EEG resources (EMBARC, CAN-BIND) are **hard-gated** (NIMH Data Archive Data-Use Certification requiring an institutional signing official; weeks–months). So **Layer A predicts MDD vs. HC, not treatment response.**

| Dataset | Contents | Labels | Access | Use |
|---|---|---|---|---|
| **Mumtaz 2016** (figshare 4244171) | 19-ch, 256 Hz, resting EC+EO + P300; 34 MDD / 30 HC, adults | **Clinically-diagnosed MDD vs HC** | **Direct download, CC BY 4.0** (behind an anti-bot WAF — browser works) | **PRIMARY** |
| **OpenNeuro ds003478** (Cavanagh / PRED+CT) | 64-ch, 500 Hz, EO+EC; 122 young adults | **BDI self-report cutoffs** (subclinical college sample; only ~11–21 met current-MDD) | **Direct, CC0** | **Fallback / cross-dataset robustness check** (it's a *symptom* classifier, weaker as the clinical claim) |
| **MODMA** (brief's "primary lead") | 128-ch + 3-ch wearable, 250 Hz + audio; ~53 subjects | MDD vs HC (PHQ-9/MINI); **no treatment response** | **Gated** — institutional email + signed EULA + admin approval (~days). **Confirmed: not a today-download.** | Not used (access risk for a 1-week build) |
| **TD-BRAIN** | 26-ch resting, n=1,274 (MDD=426) | rTMS/neurofeedback outcomes (not drugs) | Free but **registration + click-through DUA** (~same-day) | Optional stretch only |
| **EMBARC / CAN-BIND** | Baseline EEG + antidepressant RCT outcomes | **Treatment response** ✓ | **Hard-gated** (institutional) | Not feasible |

**Adolescent gap (honest limitation):** every openly-accessible EEG dataset is an **adult** cohort. A model trained here is applied **off-distribution** to the adolescent target users — a research-prototype caveat, not a validated tool.

**Verification gaps (explicit honesty):** the dataset scout confirmed Mumtaz is *public and downloadable* via the figshare API/DataCite but did **not** complete an actual byte-level download in-session (the WAF blocks automated fetch; a browser passes). ds003478 channel/rate specs come from corroborating papers, not the live JS page. → **PLAN.md treats "verify the download works" as Day-1 task #1, with a manual-download fallback.**

**Hormonal data:** no openly-downloadable cortisol + MDD case-control set exists for a quick demo (MIDUS via ICPSR is the one real option but needs a free account and is an aging cohort, not MDD). → **synthetic, calibrated to CIRCORT/MIDUS**, with MIDUS named as the real-data upgrade path.

---

## 3. Technical approach (full recipe in PLAN.md)

- **Preprocessing (MNE-Python):** notch (50/60 Hz + harmonics) → bandpass 1–40 Hz (FIR, zero-phase; 1 Hz high-pass also satisfies the pre-ICA recommendation) → average reference (projector) → ICA for ocular/cardiac *only if EOG/ECG channels exist*, else threshold rejection → fixed-length epochs (2 s, 50% overlap) → peak-to-peak rejection (~150 µV).
- **Features (per-subject, mean over epochs; keep p small ≈10–30):** absolute & relative bandpower (δ/θ/α/β/γ via Welch); **frontal alpha asymmetry = ln(α-power F4) − ln(α-power F3)** (right − left; *higher = greater relative left-frontal activation*) — **sign convention is a real footgun → pin it in a unit test**; optional mean frontal alpha coherence (only if enough epochs).
- **Validation (the whole point): subject-wise CV** — `LeaveOneGroupOut` (n≈64) or `GroupKFold`, keyed by subject ID, so no subject appears in both train and test. **Every transform inside a `Pipeline`** (no global scaling/feature-selection/oversampling before CV). Report **ROC-AUC, sensitivity, specificity, confusion matrix, per-fold spread**, plus a **`DummyClassifier` floor** and **`permutation_test_score` (groups-aware)** p-value. Probability **calibration** via `CalibratedClassifierCV(sigmoid)`.
- **Signature honesty demo:** run the *same* model under random k-fold (leaky) vs. LOSO (correct) and report the accuracy drop on Mumtaz — the literature has no such head-to-head on EEG-MDD, and the inflated "98%" came from this dataset.
- **Models for tiny n:** regularized logistic regression (default, interpretable), linear SVM, shallow random forest. **No deep nets** (overfit < ~thousands of subjects; classical models stay competitive).

---

## 4. Responsible AI / regulatory framing

- **Sharpest regulatory finding:** because the tool **processes a physiological signal (EEG)**, it would generally fall under **Criterion 1** of the FDA's Clinical Decision Support guidance — i.e., it would be regulated as a **medical device**, **not** exempt "Non-Device CDS." → The prototype must be **loudly research-only and must NOT imply CDS-exemption.** (FDA CDS guidance: 2022 final, **revised final 2026-01-06** retains the four criteria and the "clinician can independently review the basis / not rely primarily on it" framing.)
- **Frameworks to design against:** IMDRF SaMD (inform vs. drive vs. treat/diagnose risk axis); FDA/HC/MHRA **Good Machine Learning Practice** (P7 human-AI team, P9 essential information) and **Transparency** principles (intended use, development, performance, logic).
- **Implemented as UX (mapped in PLAN.md):** persistent "Research prototype — not for clinical use" banner on every screen/export; **calibrated probability + confidence band**, never a bare verdict; explicit **"low confidence / insufficient evidence"** state; **auto-flag low-confidence cases** for human review; **feature-contribution** explanations in clinician-readable language; "**suggestion**," never "diagnosis/decision," with a required clinician **Confirm / Override / Request-review** action; a **model card** (dataset size, subject-wise metrics + CIs, permutation p, caveats incl. FAA-not-validated and the adolescent off-distribution gap).

---

## 5. Section 3 assumption audit

| Brief assumption | Verdict | Reason |
|---|---|---|
| Field is legitimate (precision/computational psychiatry) | **CONFIRM** | Active, funded field; real journals. |
| No validated biological taxonomy mapping to treatment | **CONFIRM** | Biotypes failed replication; specifiers don't reduce heterogeneity. |
| FAA is most-cited but modest/fragile | **CORRECT (nuance)** | Fragile ✓ (pooled d≈−0.007). But most-cited for *detection*; for *treatment response* the leading candidate is **rACC theta**, not FAA. |
| "98% MDD vs control" reflects overfitting/leakage | **CONFIRM** | Real in print (Mumtaz n=63) but small-n + record-wise leakage; >2.3× error inflation; ~79% at n=213. |
| Cortisol = risk factor more than state marker (adolescents) | **CONFIRM (nuance)** | True for morning cortisol/CAR; nocturnal cortisol does track current illness. |
| Both elevated & blunted cortisol linked to MDD | **CONFIRM** | Non-monotonic; moderated by subtype/sex/puberty. |
| Melancholic vs atypical differ in HPA but not a clinical test | **CONFIRM** | Group-level direction ✓; individual subtyping fails (overlap; NESDA; DST history). |
| Pharmacogenomics contested; null in adolescents | **CONFIRM (both)** | GUIDED missed primary endpoint; Vande Voort 2022 (n=176) negative. Caveat: single-gene PK ≠ combinatorial panels. |
| Bespoke "custom medicine per patient" infeasible | **CONFIRM** | Only selection/dosing of *existing* drugs is plausible — and even that is contested. |
| Treatment-response-labeled open data is scarce | **CONFIRM** | None openly downloadable; EMBARC/CAN-BIND hard-gated. |

**Net:** every Section 3 assumption holds; the only substantive correction is that **rACC theta (not FAA) is the leading treatment-response EEG candidate** — and it is prognostic, not prescriptive.

---

## 6. Implications for scope (feeds PLAN.md)

1. **Layer A is real and honest, but modest by design:** MDD vs. HC on Mumtaz, subject-wise CV, honest metrics — *plus* the k-fold-vs-LOSO leakage demonstration as the credibility centerpiece.
2. **Layer B is explicitly illustrative:** synthetic patients; treatment-response/subtype/hormonal panels framed as "how this *would* work," never as real model outputs; uncertainty and clinician control front and center.
3. **Claims ceiling:** detection-pipeline-is-real; *no* treatment-prediction, *no* clinical validation, *no* CDS-exemption, *not* validated in adolescents.
4. **The honesty *is* the product.** In a field full of overclaiming demos, a correctly-validated modest result + a visible leakage demonstration + disciplined uncertainty will read as more credible to informed evaluators than any "99% cure-picker."

---

## 7. Sources

**EEG biomarkers / ML validation**
- van der Vinne et al. (2017), *NeuroImage: Clinical* — FAA meta-analysis (d≈−0.007). https://pmc.ncbi.nlm.nih.gov/articles/PMC5524421/
- npj Mental Health Research (2025) — FAA meta-analysis (d≈−0.03). https://pmc.ncbi.nlm.nih.gov/articles/PMC11739517/
- eLife (2021) — FAA multiverse analysis. https://elifesciences.org/articles/60595
- iSPOT-D / Arns et al. (2016) — FAA predicts SSRI response in women. https://pubmed.ncbi.nlm.nih.gov/26189209/
- Pizzagalli et al. (2001), *AJP* — rACC theta (n=18). https://psychiatryonline.org/doi/10.1176/appi.ajp.158.3.405
- Pizzagalli et al. (2018), *JAMA Psychiatry* (EMBARC) — rACC theta prognostic, treatment-nonspecific. https://pubmed.ncbi.nlm.nih.gov/29641834/
- Whitton et al. (2019), *Biol Psychiatry* — EMBARC confirmation. https://pubmed.ncbi.nlm.nih.gov/30718038/
- Mumtaz et al. (2018) — 98% SVM, n=63. https://pubmed.ncbi.nlm.nih.gov/28702811/
- Saeb et al. (2017), *GigaScience* — record-wise vs subject-wise CV. https://academic.oup.com/gigascience/article/6/5/gix019/3071704
- Brookshire et al. (2024) — data leakage in translational EEG (99.8%→53%). https://doi.org/10.3389/fnins.2024.1373515
- Poldrack, Huckins, Varoquaux (2020), *JAMA Psychiatry* — small-n prediction. https://pmc.ncbi.nlm.nih.gov/articles/PMC7250718/
- Kriegeskorte et al. (2009), *Nat Neurosci* — circular analysis / double dipping. https://www.nature.com/articles/nn.2303

**Biotypes / subtypes**
- Drysdale et al. (2017), *Nature Medicine* — 4 biotypes. https://www.nature.com/articles/nm.4246
- Dinga et al. (2019), *NeuroImage: Clinical* — failed replication. https://pmc.ncbi.nlm.nih.gov/articles/PMC6543446/
- STAR\*D specifier analysis (2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8447832/

**Hormonal / HPA**
- Zajkowska et al. (2022), *Psychoneuroendocrinology* — cortisol & adolescent depression meta-analysis. https://pmc.ncbi.nlm.nih.gov/articles/PMC8783058/
- Nandam et al. (2020), *Front. Psychiatry* — cortisol & MDD. https://www.frontiersin.org/articles/10.3389/fpsyt.2019.00974/full
- Juruena et al. (2018), *J. Affective Disorders* — atypical vs non-atypical HPA. https://pubmed.ncbi.nlm.nih.gov/29150144/
- Stetler & Miller (2011), *Psychosomatic Medicine* — HPA meta-summary. https://pubmed.ncbi.nlm.nih.gov/21257974/
- Lamers et al. (2013), *Mol. Psychiatry* (NESDA). https://www.nature.com/articles/mp2012144
- Beijers et al. (2019), *Psychological Medicine* (NESDA LCA). https://pmc.ncbi.nlm.nih.gov/articles/PMC6393228/
- APA Task Force (1987), *AJP* — DST overview. https://pubmed.ncbi.nlm.nih.gov/3310667/

**Pharmacogenomics**
- Greden et al. / GUIDED (2019), *J. Psychiatr. Res.* — primary endpoint p=.069. https://www.sciencedirect.com/science/article/pii/S0022395618310069
- Tesfamicael et al. (2024), *Front. Psychiatry* — umbrella review + meta-analysis. https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2024.1276410/full
- Vande Voort et al. (2022), *JAACAP* — adolescent RCT (negative). https://pubmed.ncbi.nlm.nih.gov/34099307/
- FDA safety communication / warning (2018–2019). https://www.fda.gov/news-events/press-announcements/fda-issues-warning-letter-genomics-lab-illegally-marketing-genetic-test-claims-predict-patients
- APA Council of Research (2024), *AJP*. https://psychiatryonline.org/doi/full/10.1176/appi.ajp.20230657

**Methods / regulatory**
- MNE-Python docs (filtering, ICA, PSD, epochs). https://mne.tools/stable/
- scikit-learn docs (GroupKFold/LeaveOneGroupOut, Pipeline, permutation_test_score, calibration). https://scikit-learn.org/stable/modules/cross_validation.html
- FDA, Clinical Decision Support Software guidance (2022 final; revised final 2026-01-06). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- IMDRF, SaMD key definitions & risk framework (2013/2014). https://www.imdrf.org/documents/software-medical-device-samd-key-definitions
- FDA/HC/MHRA, Good Machine Learning Practice (2021) & Transparency principles (2024). https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles

*Verification note: high-stakes specifics (GUIDED p=.069, Dinga null clusters, Vande Voort negative, EMBARC treatment-nonspecific, >2.3× CV-leakage inflation, Mumtaz public-download status) were each cross-checked against primary sources. A few paywalled exacts (Pizzagalli 2001 effect size, some second-decimal ML accuracies) are reported at abstract/review precision and flagged as such.*
