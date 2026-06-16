# Data access

Layer A trains on **open, licensed** EEG data. **Raw EEG is not committed** (license + size); `data/raw/` is git-ignored. Download it yourself with the steps below.

## Primary: Mumtaz 2016 — MDD vs. Healthy-Control resting EEG

- **Source:** figshare, "EEG Data New" — https://figshare.com/articles/dataset/EEG_Data_New/4244171
- **DOI:** 10.6084/m9.figshare.4244171
- **License:** CC BY 4.0 (attribution required; we keep raw data out of git regardless)
- **Contents:** 19-channel (10–20), 256 Hz resting EEG; ~34 MDD + ~30 HC adults; eyes-closed (EC), eyes-open (EO), and a task condition. EDF files named by group (`MDD`/`H`) + subject + condition (e.g. `MDD S1 EC.edf`).

### Steps
1. Open the figshare page above **in a browser** (an automated download is blocked by figshare's anti-bot WAF; a browser works fine).
2. Click **Download all** (zip), or download the individual `.edf` files.
3. Unzip and place all `.edf` files here:
   ```
   data/raw/mumtaz/
   ```
4. Verify: `ls data/raw/mumtaz/*.edf | wc -l` → a few dozen+ files.

Then run the Layer A pipeline (see README → `ml/train.py`).

## Fallback: OpenNeuro ds003478 (CC0)
- https://openneuro.org/datasets/ds003478 — 64-ch, 122 young adults, **BDI-defined (subclinical)** labels. Frictionless, but a weaker clinical claim; used only as a robustness check. Place under `data/raw/ds003478/`.

## Not used
- **MODMA** — gated (institutional email + signed EULA + admin approval); excluded for a 1-week build.
- **EMBARC / CAN-BIND** — the only open-ish *treatment-response* EEG, but hard-gated (NIMH Data Archive DUC). Out of scope.

## Hormonal data
Cortisol inputs in Layer B are **synthetic**, calibrated to published reference distributions (CIRCORT / MIDUS). No real patient hormonal data is used. (MIDUS via ICPSR is the documented real-data upgrade path.)
