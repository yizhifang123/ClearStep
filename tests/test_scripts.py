import numpy as np
import pytest

from ml.robustness import features_for_condition
from ml.transfer_eval import evaluate_transfer


def test_robustness_empty_condition_has_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="No EDF files for condition=EC"):
        features_for_condition(tmp_path, "EC")


class _DummyModel:
    def predict_proba(self, X):
        X = np.asarray(X)
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


def test_transfer_evaluation_requires_usable_labeled_recordings():
    with pytest.raises(SystemExit, match="No labeled usable recordings"):
        evaluate_transfer(_DummyModel(), ["feature_a"], [], {})
