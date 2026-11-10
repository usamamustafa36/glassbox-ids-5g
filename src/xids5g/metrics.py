"""Evaluation metrics, with the minority-class picture always made explicit."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
)

from . import config as C


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray,
                  y_prob: np.ndarray | None = None) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_attack": float(p),
        "recall_attack": float(r),
        "f1_attack": float(f),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def multiclass_scores(y_true: np.ndarray, y_pred: np.ndarray,
                      classes: Sequence[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(classes), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      labels=list(classes), zero_division=0)),
        "minority_macro_recall": float(minority_macro_recall(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray,
                    classes: Sequence[str]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(classes), zero_division=0)
    return pd.DataFrame({
        "class": list(classes),
        "support": s,
        "share_%": np.round(100 * s / max(1, s.sum()), 4),
        "precision": np.round(p, 4),
        "recall": np.round(r, 4),
        "f1": np.round(f, 4),
        "is_minority": [c in C.MINORITY_CLASSES for c in classes],
    })


def minority_macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean recall over the rare attack classes only."""
    present = [c for c in C.MINORITY_CLASSES if (y_true == c).any()]
    if not present:
        return float("nan")
    _, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0)
    return float(np.mean(r))


def confusion(y_true: np.ndarray, y_pred: np.ndarray,
              classes: Sequence[str], normalize: bool = False) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    return cm


# --------------------------------------------------------------------------
# Irreducible error / attainable ceiling
# --------------------------------------------------------------------------
def row_keys(X: pd.DataFrame) -> np.ndarray:
    """One hash per row, identifying byte-for-byte identical feature vectors.

    Hashing the frame directly is vectorised; building a joined string per row
    first is ~50x slower and matters on the full 1.2 million-flow dataset.
    """
    return pd.util.hash_pandas_object(X, index=False).values


def bayes_ceiling(X: pd.DataFrame, y: np.ndarray,
                  key: np.ndarray | None = None) -> dict[str, float]:
    """Best accuracy any model could reach on this feature representation.

    Rows whose feature vectors are byte-for-byte identical but whose labels
    differ cannot be told apart by *any* classifier. Predicting the majority
    label inside each identical-feature group is therefore the optimum, and its
    accuracy is an upper bound on every model trained on these features.

    ``key`` may be supplied to reuse a hash computed once by the caller.
    """
    if key is None:
        key = row_keys(X)
    frame = pd.DataFrame({"k": key, "y": np.asarray(y).astype(str)})

    # Count (group, label) pairs once, then take the largest count per group.
    # This is the majority-label count, computed without a Python callback per
    # group: on the full dataset that difference is 2.7 s against 322 s.
    pair_counts = frame.groupby(["k", "y"], sort=False).size()
    majority = pair_counts.groupby(level=0, sort=False).max()
    labels_per_key = pair_counts.groupby(level=0, sort=False).size()

    n = int(len(frame))
    ceiling = float(majority.sum() / n)
    conflicting_keys = labels_per_key.index[labels_per_key > 1]
    conflicting = frame["k"].isin(conflicting_keys)
    return {
        "ceiling_accuracy": ceiling,
        "irreducible_error": 1.0 - ceiling,
        "n_rows": n,
        "n_distinct_feature_vectors": int(len(majority)),
        "n_rows_in_conflicting_groups": int(conflicting.sum()),
        "pct_rows_in_conflicting_groups": float(100 * conflicting.mean()),
    }


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------
def scores_to_rows(name: str, scores: dict[str, float],
                   keys: Iterable[str]) -> list:
    return [name] + [scores.get(k, float("nan")) for k in keys]


def df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Markdown table without requiring the optional `tabulate` dependency."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
        else:
            d[c] = d[c].astype(str)
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(r) + " |" for r in d.astype(str).values]
    return "\n".join([header, sep, *rows])
