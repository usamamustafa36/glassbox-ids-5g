"""Model construction and persistence.

One place decides how every model is built, so "identical splits, identical
settings" is enforced by construction rather than by discipline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C

#: Which family each model belongs to. Drives the glass-box / black-box split
#: that the whole study rests on.
MODEL_FAMILY = {
    "DecisionTree": "glass-box (readable, baseline)",
    "LogisticRegression": "glass-box (readable, baseline)",
    "KNN": "black-box (instance-based)",
    "RandomForest": "black-box",
    "XGBoost": "black-box",
    "EBM": "glass-box (intrinsically interpretable)",
}


def make_ebm(task: str = "binary", *, seed: int | None = None,
             feature_types: list[str] | None = None, **overrides: Any):
    from interpret.glassbox import ExplainableBoostingClassifier

    params = dict(C.EBM_PARAMS)
    params["interactions"] = (C.EBM_BINARY_INTERACTIONS if task == "binary"
                              else C.EBM_MULTI_INTERACTIONS)
    if seed is not None:
        params["random_state"] = seed
    params.update(overrides)
    return ExplainableBoostingClassifier(
        n_jobs=C.N_JOBS_EBM, feature_types=feature_types, **params)


def make_xgb(*, seed: int | None = None, n_classes: int = 2, **overrides: Any):
    import xgboost as xgb

    params = dict(C.XGB_MULTI_PARAMS if n_classes > 2 else C.XGB_PARAMS)
    if seed is not None:
        params["random_state"] = seed
    params.update(overrides)
    params.setdefault("n_jobs", C.N_JOBS)
    if n_classes > 2:
        params.setdefault("objective", "multi:softprob")
    return xgb.XGBClassifier(**params)


def _scaled(estimator) -> Pipeline:
    """Scale-sensitive models get imputation + standardisation of their own.

    The rows and the features are identical to every other model; only the
    numeric conditioning differs, which is what these estimators require.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", estimator),
    ])


def build_baselines(n_classes: int = 2) -> dict[str, Any]:
    """The classifier family used as baselines in the 5G-NIDD literature."""
    return {
        "DecisionTree": DecisionTreeClassifier(**C.DT_PARAMS),
        "LogisticRegression": _scaled(LogisticRegression(**C.LR_PARAMS)),
        "KNN": _scaled(KNeighborsClassifier(n_jobs=C.N_JOBS, **C.KNN_PARAMS)),
        "RandomForest": RandomForestClassifier(n_jobs=C.N_JOBS, **C.RF_PARAMS),
    }


def subsample_for_knn(X: pd.DataFrame, y: np.ndarray,
                      seed: int = C.SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Cap the k-NN reference set; prediction cost is linear in its size."""
    if len(X) <= C.KNN_MAX_TRAIN:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=C.KNN_MAX_TRAIN, replace=False)
    return X.iloc[idx], np.asarray(y)[idx]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def model_path(profile: str, name: str) -> Path:
    d = C.ARTIFACTS_DIR / profile / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.joblib"


def save_model(model: Any, profile: str, name: str) -> Path:
    import joblib

    p = model_path(profile, name)
    joblib.dump(model, p, compress=3)
    return p


def load_model(profile: str, name: str) -> Any:
    import joblib

    p = model_path(profile, name)
    if not p.exists():
        raise FileNotFoundError(
            f"Model '{name}' has not been trained for profile '{profile}'.\n"
            f"Expected: {p}\nRun the earlier pipeline stages first "
            f"(see DEMO_GUIDE.md)."
        )
    return joblib.load(p)


def has_model(profile: str, name: str) -> bool:
    return model_path(profile, name).exists()
