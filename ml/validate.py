"""Layer A — subject-wise validation, honest metrics, and the leakage demo.
NOT for clinical use.

Why subject-wise: random k-fold over epoch rows from the same subject leaks that
subject's EEG "fingerprint" into the test fold and inflates accuracy (RESEARCH.md
§3; Saeb 2017; Brookshire 2024). `leakage_demo` makes that gap visible by running
the SAME model under leaky vs. grouped CV.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(C: float = 1.0) -> Pipeline:
    """Median-impute -> standardize -> L2 logistic regression (class-balanced).

    Every transform lives INSIDE the pipeline so it is refit per CV fold — test
    statistics never leak into preprocessing.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        # L2 is the default; the explicit `penalty=` arg is deprecated in
        # scikit-learn 1.8+, so we omit it and control strength via C.
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000)),
    ])


def _stratified_cv_for_labels(ys, requested_splits: int = 5, random_state: int = 0):
    ys = np.asarray(ys, int)
    classes, counts = np.unique(ys, return_counts=True)
    if len(classes) != 2 or counts.min() < 2:
        raise ValueError("Calibration and stratified permutation require at least two subjects per class.")
    n_splits = min(int(requested_splits), int(counts.min()))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def calibration_cv_for_labels(ys, requested_splits: int = 5, random_state: int = 0):
    """Return a stratified calibration CV that cannot exceed the smallest class."""
    return _stratified_cv_for_labels(ys, requested_splits, random_state)


def _sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec, (int(tn), int(fp), int(fn), int(tp))


def _metric_values(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    sens, spec, _cm = _sens_spec(y_true, pred)
    return {
        "auc_loso": float(roc_auc_score(y_true, proba)),
        "accuracy_loso": float(accuracy_score(y_true, pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
    }


def bootstrap_metric_intervals(y_true, proba, n_bootstraps: int = 1000,
                               random_state: int = 0) -> dict:
    """Subject-level percentile bootstrap intervals for reported LOSO metrics."""
    y_true = np.asarray(y_true, int)
    proba = np.asarray(proba, float)
    rng = np.random.default_rng(random_state)
    values = {k: [] for k in ["auc_loso", "accuracy_loso", "sensitivity", "specificity"]}

    for _ in range(int(n_bootstraps)):
        idx = rng.integers(0, len(y_true), len(y_true))
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        mb = _metric_values(yb, proba[idx])
        for key, value in mb.items():
            if np.isfinite(value):
                values[key].append(value)

    intervals = {}
    for key, vals in values.items():
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            intervals[key] = [float(lo), float(hi)]
        else:
            intervals[key] = [float("nan"), float("nan")]
    return intervals


def honest_subject_evaluation(Xs, ys, C: float = 1.0, n_permutations: int = 1000,
                              random_state: int = 0, n_bootstraps: int = 1000,
                              n_jobs: int = 1) -> dict:
    """Leave-one-subject-out evaluation on per-subject feature rows.

    Honest metrics + a majority-class floor + a permutation-test p-value.
    """
    Xs = np.asarray(Xs, float)
    ys = np.asarray(ys, int)
    pipe = make_pipeline(C)

    proba = cross_val_predict(pipe, Xs, ys, cv=LeaveOneOut(),
                              method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    sens, spec, cm = _sens_spec(ys, pred)
    permutation_cv = _stratified_cv_for_labels(ys, requested_splits=5,
                                               random_state=random_state)

    score, _perm, pvalue = permutation_test_score(
        pipe, Xs, ys, scoring="roc_auc",
        cv=permutation_cv,
        n_permutations=n_permutations, random_state=random_state, n_jobs=n_jobs,
    )
    majority = float(max(ys.mean(), 1 - ys.mean()))
    ci = bootstrap_metric_intervals(ys, proba, n_bootstraps=n_bootstraps,
                                    random_state=random_state)
    return {
        "n_subjects": int(len(ys)),
        "n_mdd": int(ys.sum()),
        "auc_loso": float(roc_auc_score(ys, proba)),
        "accuracy_loso": float(accuracy_score(ys, pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "confusion_tn_fp_fn_tp": list(cm),
        "permutation_auc": float(score),
        "permutation_pvalue": float(pvalue),
        "permutation_cv": (
            f"StratifiedKFold(n_splits={permutation_cv.n_splits}, "
            f"shuffle=True, random_state={random_state})"
        ),
        "permutation_note": (
            "Permutation p-value uses stratified k-fold ROC-AUC because ROC-AUC "
            "cannot be fold-scored under LeaveOneOut test folds of one subject."
        ),
        "metric_ci_95": ci,
        "majority_class_accuracy": majority,
        "chance_auc": 0.5,
    }


def leakage_demo(X, y, groups, n_splits: int = 5, C: float = 1.0,
                 random_state: int = 0) -> dict:
    """Same model, two CV schemes on per-EPOCH rows.

    leaky   = random StratifiedKFold (a subject's epochs bleed across folds)
    grouped = GroupKFold by subject (honest)
    The accuracy/AUC gap is the demonstration.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    pipe = make_pipeline(C)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    leaky = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]

    gkf = GroupKFold(n_splits=n_splits)
    grouped = cross_val_predict(pipe, X, y, cv=gkf, groups=groups,
                                method="predict_proba")[:, 1]

    def acc(p):
        return float(accuracy_score(y, (p >= 0.5).astype(int)))

    return {
        "leaky_accuracy": acc(leaky),
        "leaky_auc": float(roc_auc_score(y, leaky)),
        "grouped_accuracy": acc(grouped),
        "grouped_auc": float(roc_auc_score(y, grouped)),
        "delta_accuracy": acc(leaky) - acc(grouped),
        "n_epochs": int(len(y)),
        "n_subjects": int(len({*groups.tolist()})),
    }
