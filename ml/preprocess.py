"""Layer A — EEG preprocessing (MNE-Python). NOT for clinical use.

Resting-state pipeline (recipe justified in RESEARCH.md §3):
  notch -> band-pass 1-40 Hz -> average reference -> [ICA iff EOG/ECG present]
  -> fixed-length epochs (2 s, 50% overlap) -> peak-to-peak rejection.

Mumtaz 2016 was recorded in Malaysia => mains is 50 Hz (notch 50/100 by default).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import mne

log = logging.getLogger(__name__)

LABELS = {"HC": 0, "MDD": 1}  # positive class = MDD (the clinical "case")

# MNE's standard_1020 montage uses modern temporal/parietal names (T7/T8/P7/P8).
_OLD_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Standard 10-20 SCALP electrodes we keep. Everything else (e.g. the Mumtaz
# 'A2-A1' linked-ear reference and '23A-23R'/'24A-24R' auxiliary channels) is
# dropped so it can't contaminate the average reference or global features.
SCALP_1020 = {
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
}


def clean_channel_name(name: str) -> str:
    """Map raw EDF channel labels to standard 10-20 names.

    Mumtaz channels look like 'EEG Fp1-LE' (linked-ear reference). Strip the
    'EEG ' prefix and a trailing reference token, then modernise T3/T4/T5/T6.
    """
    n = re.sub(r"^\s*EEG\s*", "", name, flags=re.IGNORECASE)
    n = re.sub(r"[-_ ]?(LE|RE|REF|A1A2|A2A1|M1M2|AVG|Ref)\s*$", "", n, flags=re.IGNORECASE)
    n = n.strip()
    return _OLD_TO_MODERN.get(n, n)


def process_raw(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch=(50.0, 100.0),
    epoch_len: float = 2.0,
    overlap: float = 1.0,
    reject_uv: float = 150.0,
    do_ica: bool = False,
) -> mne.Epochs:
    """Clean a loaded Raw and return fixed-length epochs.

    Split out from file I/O so the MNE pipeline is testable on a synthetic Raw.
    """
    raw.rename_channels({c: clean_channel_name(c) for c in raw.ch_names})
    raw.pick([c for c in raw.ch_names if c in SCALP_1020])  # scalp electrodes only
    raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")

    if notch:
        raw.notch_filter(freqs=list(notch), verbose="ERROR")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    if do_ica:
        has_eog = len(mne.pick_types(raw.info, eog=True)) > 0
        has_ecg = len(mne.pick_types(raw.info, ecg=True)) > 0
        if has_eog or has_ecg:
            ica = mne.preprocessing.ICA(n_components=15, method="picard",
                                        max_iter="auto", random_state=97, verbose="ERROR")
            ica.fit(raw)
            bads: list[int] = []
            if has_eog:
                bads += ica.find_bads_eog(raw, verbose="ERROR")[0]
            if has_ecg:
                bads += ica.find_bads_ecg(raw, verbose="ERROR")[0]
            ica.exclude = sorted(set(bads))
            ica.apply(raw, verbose="ERROR")
        else:
            log.info("do_ica=True but no EOG/ECG channels found; skipping ICA.")

    epochs = mne.make_fixed_length_epochs(
        raw, duration=epoch_len, overlap=overlap, preload=True, verbose="ERROR")
    epochs.drop_bad(reject={"eeg": reject_uv * 1e-6}, verbose="ERROR")
    return epochs


def load_and_preprocess(edf_path, **kwargs) -> mne.Epochs:
    """Read one EDF file and return cleaned fixed-length epochs."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    return process_raw(raw, **kwargs)


def discover_mumtaz(root) -> list[dict]:
    """Scan data/raw/mumtaz for EDFs; infer (subject, label, condition).

    Mumtaz filenames look like 'MDD S1 EC.edf' / 'H S2 EO.edf'. This parser is
    defensive; the exact naming is verified against the real download on Day 1.
    """
    root = Path(root)
    records: list[dict] = []
    for p in sorted(root.glob("*.edf")):
        stem = p.stem.upper()
        if "MDD" in stem:
            label = "MDD"
        elif re.search(r"(^|[^A-Z])H([^A-Z]|$)|HEALTHY|CONTROL|\bHC\b", stem):
            label = "HC"
        else:
            log.warning("Could not infer label for %s; skipping.", p.name)
            continue
        m = re.search(r"S\s*0*(\d+)", stem)
        subject = f"{label}_S{m.group(1)}" if m else f"{label}_{p.stem}"
        cond = "EC" if "EC" in stem else "EO" if "EO" in stem else "TASK"
        records.append({"path": str(p), "subject": subject,
                        "label": label, "condition": cond})
    return records
