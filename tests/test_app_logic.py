import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import model_io
import synthetic


def test_synthetic_patient_structure_and_determinism():
    p1 = synthetic.generate_synthetic_patient(7, "melancholic_leaning")
    p2 = synthetic.generate_synthetic_patient(7, "melancholic_leaning")
    assert p1.to_dict() == p2.to_dict()                       # deterministic by seed
    assert p1.sex == "male" and 14 <= p1.age <= 18
    assert set(p1.eeg.keys()) == set(synthetic.EEG_FEATURE_NAMES)
    assert {"morning_nmol_l", "car_increase_pct", "evening_nmol_l",
            "diurnal_slope"} <= set(p1.cortisol)
    frontal = [p1.eeg[f"rel_{b}_frontal"] for b in synthetic.BAND_ORDER]
    assert abs(sum(frontal) - 1.0) < 1e-6                      # relative powers sum to 1


def test_subtype_leaning_monotonic_and_normalized():
    low = synthetic.illustrative_subtype_leaning({"morning_nmol_l": 5.0})
    high = synthetic.illustrative_subtype_leaning({"morning_nmol_l": 25.0})
    assert abs(low["melancholic"] + low["atypical"] - 1.0) < 1e-9
    assert high["melancholic"] > low["melancholic"]           # higher cortisol -> melancholic


def test_feature_contributions_sorted_by_magnitude():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 4)
    y = (X[:, 0] + rng.randn(40) * 0.1 > 0).astype(int)
    pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                     ("clf", LogisticRegression())]).fit(X, y)
    contribs = model_io.feature_contributions(pipe, ["a", "b", "c", "d"], X[0])
    assert len(contribs) == 4
    mags = [abs(c) for _, c in contribs]
    assert mags == sorted(mags, reverse=True)
