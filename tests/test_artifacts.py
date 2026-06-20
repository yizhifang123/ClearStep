import pandas as pd
import pytest
from pathlib import Path
import json
import joblib

from ml.train import artifact_metadata, build_feature_reference

ROOT = Path(__file__).resolve().parents[1]


def test_build_feature_reference_records_training_distribution():
    df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0, 40.0]})

    ref = build_feature_reference(df, ["a", "b"])

    assert ref["a"]["min"] == 0.0
    assert ref["a"]["max"] == 3.0
    assert ref["a"]["p01"] == pytest.approx(0.03)
    assert ref["a"]["p99"] == pytest.approx(2.97)
    assert ref["b"]["mean"] == pytest.approx(25.0)


def test_artifact_metadata_records_schema_training_and_runtime_context():
    meta = artifact_metadata(
        condition="EC",
        data_root="data/raw/mumtaz",
        feature_names=["a", "b"],
        preprocessing={"l_freq": 1.0, "h_freq": 40.0},
    )

    assert meta["schema_version"] == 2
    assert meta["training"]["condition"] == "EC"
    assert meta["training"]["data_root"] == "data/raw/mumtaz"
    assert meta["training"]["n_features"] == 2
    assert meta["preprocessing"] == {"l_freq": 1.0, "h_freq": 40.0}
    assert "generated_at_utc" in meta
    assert "python" in meta["runtime"]


def test_committed_metrics_include_schema_and_interval_metadata():
    metrics = json.loads((ROOT / "ml/artifacts/metrics.json").read_text())

    assert metrics["schema_version"] == 2
    assert "metadata" in metrics
    assert "metric_ci_95" in metrics["honest_loso"]
    assert "permutation_cv" in metrics["honest_loso"]


def test_committed_model_bundle_includes_reference_ranges():
    bundle = joblib.load(ROOT / "ml/artifacts/model.pkl")

    assert bundle["schema_version"] == 2
    assert "metadata" in bundle
    assert set(bundle["feature_names"]) <= set(bundle["feature_reference"])
