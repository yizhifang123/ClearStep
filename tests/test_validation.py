import numpy as np
from sklearn.model_selection import GroupKFold

from ml.validate import leakage_demo


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
