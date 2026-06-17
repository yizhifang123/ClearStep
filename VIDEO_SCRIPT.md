# VIDEO_SCRIPT.md — ~2:45 demo recording script

Record with the app running (`​.venv/bin/streamlit run app/Home.py`). Narration in quotes; on-screen action in *italics*. Aim for calm and honest — the restraint is the selling point.

---

**[0:00–0:20] Hook — the problem**
*On screen: the Home page banner + title.*
> "Depression treatment is trial-and-error — patients endure several failed medications before relief. We built a decision-*support* prototype that explores whether brain signals could help personalize it. The catch: the science isn't fully there yet — so we built it honestly."

**[0:20–0:40] The two layers**
*On screen: Home page, the real-vs-illustrative table.*
> "Two layers. Layer A is a real machine-learning pipeline on open EEG data. Layer B is an illustrative clinician dashboard. We label exactly what's real and what's simulated — that honesty is the point."

**[0:40–1:20] The credibility moment — Model Card**
*Navigate to Model Card. Show the metrics, then scroll to the warning + the beta-driver table.*
> "On 58 subjects, correctly validated subject-wise, we get AUC 0.95. That sounds like the inflated numbers you see in this field — so we dug in. It's driven almost entirely by beta-band power, which can be muscle artifact as easily as depression, on a small single-site adult dataset. And frontal alpha asymmetry — the famous biomarker — contributes nothing, which actually matches the research that it's near-null. So: a high number, reported with the reasons you shouldn't trust it clinically."

**[1:20–1:45] The leakage demonstration**
*Scroll to the leakage figure.*
> "We also demonstrate the field's most common mistake: validate the leaky way and accuracy inflates. We validate subject-wise — same model, honest split — and show the gap."

**[1:45–2:15] Real case + human-in-the-loop**
*Navigate to Real Case. Pick a subject. Point at the probability, confidence state, and the contribution chart, then the Confirm/Override buttons.*
> "For a held-out patient, the tool shows a calibrated probability — not a verdict — with the features that drove it. Low-confidence cases are flagged for review. And the clinician always decides: confirm, override, or escalate. The AI only suggests."

**[2:15–2:40] Illustrative workflow + honest limits**
*Navigate to Illustrative Case. Show the synthetic cortisol panel and subtype leaning, then the "what this does NOT do" section.*
> "Here's the envisioned workflow on a synthetic patient — hormones, a subtype leaning — all clearly simulated. Notice what it refuses to do: pick or dose a drug. The evidence doesn't support that, and pharmacogenomic drug-pickers failed in an adolescent trial."

**[2:40–2:45] Close**
*Back to the banner.*
> "Research prototype, not for clinical use. Honest about what's real — and what isn't."

---

### Capture checklist
- [ ] App running locally with trained artifacts present (Home shows AUC 0.95)
- [ ] 1080p screen capture; hide bookmarks bar
- [ ] Pre-pick a clear Real Case subject and the "Case A — melancholic" Illustrative case
- [ ] Keep the "not for clinical use" banner visible in at least the first and last shots
