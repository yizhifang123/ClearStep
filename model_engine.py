"""The real ML core (Layer A): loads the trained EEG classifier and runs it on a
synthetic patient's feature vector.

This is genuine AI capability evidence: a logistic-regression pipeline trained on
the public Mumtaz EEG dataset with leave-one-subject-out validation (AUC ~0.95,
p<0.001). We apply it to SYNTHETIC inputs only — never real patients.

The engine adds the honesty layer the raw model lacks: a calibrated probability,
an explicit low-confidence band, and out-of-distribution flags when a synthetic
feature falls outside the range the model actually saw in training.
"""

import warnings
from pathlib import Path

import joblib

warnings.filterwarnings("ignore")  # benign sklearn cross-version unpickle notices

from synthetic import (  # noqa: E402  (import after warnings filter)
    CURATED_CASES,
    generate_synthetic_patient,
    illustrative_subtype_leaning,
)

HERE = Path(__file__).parent
LOW_CONF = (0.40, 0.60)  # probabilities in this band are flagged "uncertain"

_BUNDLE = None


def _bundle():
    # Safe: model.pkl is our OWN trained artifact, produced by this project's
    # ml/train.py and shipped in this repo — not user-supplied or untrusted input.
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(HERE / "model.pkl")
    return _BUNDLE


def metrics():
    """Validation metrics (with 95% CIs) for the Model Card / clinician view."""
    return _bundle()["metrics"]["honest_loso"]


def confidence_label(prob):
    if LOW_CONF[0] <= prob <= LOW_CONF[1]:
        return "uncertain"
    return "higher" if (prob > LOW_CONF[1] or prob < LOW_CONF[0]) else "moderate"


def _ood_flags(eeg):
    """Flag synthetic features outside the model's 1st-99th training percentile."""
    ref = _bundle()["feature_reference"]
    flags = []
    for name, value in eeg.items():
        r = ref.get(name)
        if not r:
            continue
        lo, hi = r.get("p01"), r.get("p99")
        if lo is not None and hi is not None and not (lo <= value <= hi):
            flags.append(f"{name} = {value:.3f} (training range {lo:.3f}–{hi:.3f})")
    return flags


def analyze_patient(seed, archetype):
    """Run the full Layer A analysis on a synthetic patient. No API key needed."""
    bundle = _bundle()
    feats, clf = bundle["feature_names"], bundle["calibrated"]
    patient = generate_synthetic_patient(seed, archetype)

    prob = float(clf.predict_proba([patient.eeg_vector(feats)])[0][1])
    subtype = illustrative_subtype_leaning(patient.cortisol)

    return {
        "patient": patient.to_dict(),
        "p_mdd": prob,
        "confidence": confidence_label(prob),
        "low_conf_band": LOW_CONF,
        "ood_flags": _ood_flags(patient.eeg),
        "subtype_leaning": subtype,  # illustrative, from cortisol only
        "metrics": metrics(),
    }


# Curated cases tuned for a clean demo spread (low / borderline / elevated signal).
# Each is (seed, archetype); the model output is real, computed at runtime.
DEMO_CASES = {
    "Patient 1 — elevated EEG signal, wide uncertainty": (16, "control_like"),
    "Patient 2 — borderline / inconclusive signal": (350, "control_like"),
    "Patient 3 — low EEG signal": (235, "melancholic_leaning"),
}
