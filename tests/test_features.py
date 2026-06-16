import numpy as np

from ml.features import (
    BANDS,
    FRONTAL,
    POSTERIOR,
    aggregate_by_subject,
    frontal_alpha_asymmetry,
)


def test_faa_sign_convention():
    # right alpha > left alpha  -> positive FAA -> greater relative LEFT activation
    assert frontal_alpha_asymmetry(1.0, np.e) > 0
    assert abs(frontal_alpha_asymmetry(1.0, np.e) - 1.0) < 1e-9   # exact: ln(e)-ln(1)
    assert abs(frontal_alpha_asymmetry(2.0, 2.0)) < 1e-12         # symmetric -> 0
    # left > right -> negative (greater relative RIGHT activation; depression-linked)
    assert frontal_alpha_asymmetry(np.e, 1.0) < 0


def test_bands_and_rois():
    assert list(BANDS) == ["delta", "theta", "alpha", "beta", "gamma"]
    assert BANDS["alpha"] == (8.0, 13.0)
    assert "F3" in FRONTAL and "F4" in FRONTAL
    assert "O1" in POSTERIOR and "O2" in POSTERIOR


def test_aggregate_by_subject_means_rows():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [10.0, 10.0]])
    groups = np.array(["a", "a", "b"])
    Xs, order = aggregate_by_subject(X, groups)
    assert list(order) == ["a", "b"]
    assert np.allclose(Xs[0], [1.0, 2.0])    # mean of subject a's two rows
    assert np.allclose(Xs[1], [10.0, 10.0])
