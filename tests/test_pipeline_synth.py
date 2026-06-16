"""Integration test of the MNE pipeline on a SYNTHETIC Raw (no EDF needed).

Exercises notch -> band-pass -> reference -> epoching -> rejection -> feature
extraction, so the path that `ml/train.py` runs on real data is validated before
the real download arrives.
"""
import mne
import numpy as np

from ml.features import epoch_feature_matrix
from ml.preprocess import clean_channel_name, process_raw

# Mumtaz-style channel labels (linked-ear 'LE' suffix; old temporal names).
MUMTAZ_CH = ["EEG Fp1-LE", "EEG Fp2-LE", "EEG F7-LE", "EEG F3-LE", "EEG Fz-LE",
             "EEG F4-LE", "EEG F8-LE", "EEG T3-LE", "EEG C3-LE", "EEG Cz-LE",
             "EEG C4-LE", "EEG T4-LE", "EEG T5-LE", "EEG P3-LE", "EEG Pz-LE",
             "EEG P4-LE", "EEG T6-LE", "EEG O1-LE", "EEG O2-LE"]


def _synth_raw(seconds=30, sfreq=256.0, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(MUMTAZ_CH), int(seconds * sfreq))) * 8e-6
    info = mne.create_info(MUMTAZ_CH, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_clean_channel_name_mumtaz_style():
    assert clean_channel_name("EEG Fp1-LE") == "Fp1"
    assert clean_channel_name("EEG T3-LE") == "T7"     # old -> modern montage name
    assert clean_channel_name("F4") == "F4"


def test_process_raw_to_features_shape_and_ranges():
    epochs = process_raw(_synth_raw())
    assert len(epochs) >= 5
    X, names = epoch_feature_matrix(epochs)
    assert X.shape[1] == 16 and len(names) == 16
    assert "faa_F4_F3" in names
    # FAA computable (F3/F4 present after cleaning)
    assert not np.isnan(X[:, names.index("faa_F4_F3")]).all()
    # relative-power features lie in [0, 1]
    col = X[:, names.index("rel_alpha_global")]
    col = col[~np.isnan(col)]
    assert ((col >= 0) & (col <= 1)).all()
