"""Layer A — feature extraction. NOT for clinical use.

Features are computed PER EPOCH (one row per epoch — this is what enables the
leakage demo), then aggregated to per-subject by mean for the honest model.

Feature set (deliberately small, ~16, for n≈64 with regularization; RESEARCH.md §3):
  - relative band power (delta/theta/alpha/beta/gamma) over all channels,
    a frontal ROI, and a posterior ROI
  - frontal alpha asymmetry (FAA)
Relative power = band power / total-power-across-bands, per channel.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from mne.time_frequency import psd_array_welch

POWER_EPS = 1e-12

BANDS = OrderedDict([
    ("delta", (1.0, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta", (13.0, 30.0)),
    ("gamma", (30.0, 40.0)),
])

FRONTAL = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"]
POSTERIOR = ["P7", "P3", "Pz", "P4", "P8", "O1", "O2"]


def frontal_alpha_asymmetry(alpha_left: float, alpha_right: float) -> float:
    """FAA = ln(right) - ln(left) on ABSOLUTE alpha power.

    Sign convention (pinned by tests): alpha power is INVERSELY related to
    cortical activity, so a higher / more-positive FAA means more right-alpha =>
    less right-frontal activity => greater RELATIVE LEFT-frontal activation.
    A more-negative FAA (greater relative right activation) is the
    depression-linked direction. (A minority of papers use left-right, flipping
    the sign — hence the explicit test.)
    """
    left = max(float(alpha_left), POWER_EPS)
    right = max(float(alpha_right), POWER_EPS)
    return float(np.log(right) - np.log(left))


def _abs_bandpowers(psds, freqs):
    out = OrderedDict()
    for band, (lo, hi) in BANDS.items():
        idx = (freqs >= lo) & (freqs < hi)
        out[band] = psds[..., idx].sum(axis=-1)
    return out


def epoch_feature_matrix(epochs, fmin: float = 1.0, fmax: float = 40.0):
    """Return (X, feature_names) with one feature row PER EPOCH."""
    data = epochs.get_data(copy=True)                 # (n_ep, n_ch, n_times)
    sfreq = float(epochs.info["sfreq"])
    psds, freqs = psd_array_welch(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax,
        n_per_seg=int(sfreq), verbose=False,
    )                                                 # (n_ep, n_ch, n_freqs)

    absbp = _abs_bandpowers(psds, freqs)              # band -> (n_ep, n_ch)
    total = np.sum([absbp[b] for b in BANDS], axis=0)
    rel = {b: np.divide(absbp[b], total,
                        out=np.full_like(absbp[b], np.nan), where=total > 0)
           for b in BANDS}

    chidx = {c: i for i, c in enumerate(epochs.ch_names)}

    def region(rel_b, names):
        sel = [chidx[c] for c in names if c in chidx]
        if not sel:
            return np.full(rel_b.shape[0], np.nan)
        return np.nanmean(rel_b[:, sel], axis=1)

    cols, names = [], []
    for b in BANDS:
        cols.append(np.nanmean(rel[b], axis=1))
        names.append(f"rel_{b}_global")
    for b in BANDS:
        cols.append(region(rel[b], FRONTAL))
        names.append(f"rel_{b}_frontal")
    for b in BANDS:
        cols.append(region(rel[b], POSTERIOR))
        names.append(f"rel_{b}_posterior")

    a = absbp["alpha"]                                # (n_ep, n_ch)
    if "F3" in chidx and "F4" in chidx:
        right = np.clip(a[:, chidx["F4"]], POWER_EPS, None)
        left = np.clip(a[:, chidx["F3"]], POWER_EPS, None)
        faa = np.log(right) - np.log(left)
    else:
        faa = np.full(a.shape[0], np.nan)
    cols.append(faa)
    names.append("faa_F4_F3")

    return np.column_stack(cols), names


def aggregate_by_subject(X, groups):
    """Mean feature rows within each subject. Returns (X_subj, subject_order)."""
    X = np.asarray(X, float)
    groups = np.asarray(groups)
    order = list(dict.fromkeys(groups.tolist()))      # preserve first-seen order
    rows = [np.nanmean(X[groups == g], axis=0) for g in order]
    return np.vstack(rows), np.array(order, dtype=object)
