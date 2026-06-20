import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from ml.validate import calibration_cv_for_labels, honest_subject_evaluation, leakage_demo


def test_groupkfold_never_shares_a_subject():
    groups = np.repeat(np.arange(10), 5)
    y = np.tile([0, 1], 25)
    X = np.random.RandomState(0).randn(50, 3)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_leakage_inflates_accuracy_with_subject_fingerprint():
    # Each subject gets a unique one-hot identity column. Under leaky CV a subject
    # appears in train AND test, so a linear model learns that subject's label
    # from its own column; under grouped CV the held-out subject's column is
    # all-zero in training, so the signal cannot transfer -> ~chance.
    rng = np.random.RandomState(0)
    n_subj, n_ep = 20, 25
    groups = np.repeat(np.arange(n_subj), n_ep)
    y = np.repeat(rng.randint(0, 2, n_subj), n_ep)
    X = np.zeros((len(groups), n_subj))
    X[np.arange(len(groups)), groups] = 1.0
    X += rng.randn(*X.shape) * 0.01

    res = leakage_demo(X, y, groups)
    assert res["leaky_accuracy"] > res["grouped_accuracy"] + 0.1


def test_honest_evaluation_reports_intervals_and_permutation_cv_metadata():
    rng = np.random.RandomState(2)
    y = np.array([0, 1] * 8)
    X = np.column_stack([y + rng.randn(len(y)) * 0.2, rng.randn(len(y))])

    res = honest_subject_evaluation(
        X,
        y,
        n_permutations=5,
        n_bootstraps=25,
        random_state=0,
    )

    assert "metric_ci_95" in res
    for key in ["auc_loso", "accuracy_loso", "sensitivity", "specificity"]:
        lo, hi = res["metric_ci_95"][key]
        assert 0.0 <= lo <= hi <= 1.0
    assert "permutation_cv" in res
    assert "StratifiedKFold" in res["permutation_cv"]
    assert "ROC-AUC cannot be fold-scored under LeaveOneOut" in res["permutation_note"]


def test_calibration_cv_caps_splits_at_smallest_class():
    cv = calibration_cv_for_labels(np.array([0, 0, 0, 1, 1]), requested_splits=5)
    assert cv.n_splits == 2

    with pytest.raises(ValueError, match="at least two subjects per class"):
        calibration_cv_for_labels(np.array([0, 0, 0, 1]), requested_splits=5)
