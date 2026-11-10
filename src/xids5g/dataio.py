"""Loading, cleaning, encoding, splitting and caching of the datasets.

The prepared splits are written once to ``artifacts/<profile>/`` and reused by
every later stage, which is what makes the model comparison like-for-like: all
models see exactly the same rows, the same features and the same encoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

MISSING_CATEGORY = "not-observed"
UNSEEN_CODE = -1


# --------------------------------------------------------------------------
# Raw load
# --------------------------------------------------------------------------
def load_raw(n_rows: int | None, seed: int = C.SEED) -> pd.DataFrame:
    """Read Combined.csv and take a stratified sample of ``n_rows`` flows.

    Sampling is stratified on the 9-class ``Attack Type`` label so that the
    class proportions - including the 0.09% ICMPFlood class - are preserved
    exactly at every profile size.
    """
    if not C.NIDD_CSV.exists():
        raise FileNotFoundError(
            f"5G-NIDD dataset not found at {C.NIDD_CSV}.\n"
            "Expected 'data/5g-nidd/Combined.csv'."
        )
    df = pd.read_csv(C.NIDD_CSV, low_memory=False)
    if n_rows is not None and n_rows < len(df):
        frac = n_rows / len(df)
        parts = [
            g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed)
            for _, g in df.groupby(C.LABEL_MULTI, observed=True)
        ]
        df = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
@dataclass
class Preprocessor:
    """Train-fitted preprocessing state.

    Fitting happens on the training split only; the validation and test splits
    are transformed with the training statistics, so no information leaks
    backwards from the evaluation data.
    """

    feature_names: list[str]
    categorical_features: list[str]
    category_maps: dict[str, dict[str, int]]
    medians: dict[str, float]
    indicator_columns: list[str]
    dropped_columns: list[str]
    drop_reasons: dict[str, str]

    # -- fit -----------------------------------------------------------
    @classmethod
    def fit(cls, df_train: pd.DataFrame, *, keep_leakage: bool = False) -> "Preprocessor":
        drop_reasons: dict[str, str] = {}
        work = df_train.drop(columns=C.LABEL_COLUMNS, errors="ignore")

        if not keep_leakage:
            for c in C.LEAKAGE_COLUMNS:
                if c in work.columns:
                    work = work.drop(columns=c)
                    drop_reasons[c] = "capture-order identifier (leakage)"
        else:
            if "Unnamed: 0" in work.columns:
                work = work.drop(columns="Unnamed: 0")
                drop_reasons["Unnamed: 0"] = "row index"
        for c in C.LABEL_COLUMNS:
            drop_reasons.setdefault(c, "label column")

        cats = [c for c in C.CATEGORICAL_COLUMNS if c in work.columns]
        nums = [c for c in work.columns if c not in cats]

        # Drop near-empty and constant columns.
        for c in list(nums):
            miss = work[c].isna().mean()
            if miss > C.MAX_MISSING_RATE:
                work = work.drop(columns=c)
                nums.remove(c)
                drop_reasons[c] = f"{miss:.1%} missing in training data"
            elif work[c].nunique(dropna=True) <= 1:
                work = work.drop(columns=c)
                nums.remove(c)
                drop_reasons[c] = "constant (no information)"

        # Missing-value indicators for informative missingness.
        indicators = [
            c for c in nums
            if C.MISSING_INDICATOR_MIN_RATE <= work[c].isna().mean() <= C.MAX_MISSING_RATE
        ]

        medians = {c: float(work[c].median()) if work[c].notna().any() else 0.0
                   for c in nums}

        category_maps: dict[str, dict[str, int]] = {}
        for c in cats:
            levels = (work[c].fillna(MISSING_CATEGORY).astype(str)
                      .value_counts().index.tolist())
            category_maps[c] = {lvl: i for i, lvl in enumerate(sorted(levels))}

        feature_names = nums + cats + [f"{c}__not_observed" for c in indicators]
        return cls(
            feature_names=feature_names,
            categorical_features=cats,
            category_maps=category_maps,
            medians=medians,
            indicator_columns=indicators,
            dropped_columns=sorted(drop_reasons),
            drop_reasons=drop_reasons,
        )

    # -- transform -----------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        nums = [c for c in self.medians]
        out = {}
        for c in nums:
            col = pd.to_numeric(df[c], errors="coerce")
            out[c] = col.fillna(self.medians[c]).astype(np.float32)
        for c in self.categorical_features:
            mapping = self.category_maps[c]
            col = df[c].fillna(MISSING_CATEGORY).astype(str)
            out[c] = col.map(mapping).fillna(UNSEEN_CODE).astype(np.int16)
        for c in self.indicator_columns:
            out[f"{c}__not_observed"] = df[c].isna().astype(np.int8)
        return pd.DataFrame(out, index=df.index)[self.feature_names]

    # -- persistence ---------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "category_maps": self.category_maps,
            "medians": self.medians,
            "indicator_columns": self.indicator_columns,
            "dropped_columns": self.dropped_columns,
            "drop_reasons": self.drop_reasons,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Preprocessor":
        return cls(
            feature_names=d["feature_names"],
            categorical_features=d["categorical_features"],
            category_maps={k: {kk: int(vv) for kk, vv in v.items()}
                           for k, v in d["category_maps"].items()},
            medians={k: float(v) for k, v in d["medians"].items()},
            indicator_columns=d["indicator_columns"],
            dropped_columns=d["dropped_columns"],
            drop_reasons=d["drop_reasons"],
        )

    def feature_types(self) -> list[str]:
        """``feature_types`` argument for the EBM: nominal for categoricals."""
        types = []
        for f in self.feature_names:
            if f in self.categorical_features:
                types.append("nominal")
            elif f.endswith("__not_observed"):
                types.append("nominal")
            else:
                types.append("continuous")
        return types

    def decode_category(self, feature: str, code: float) -> str:
        """Turn an encoded category code back into its original string."""
        mapping = self.category_maps.get(feature)
        if mapping is None:
            return str(code)
        inverse = {v: k for k, v in mapping.items()}
        return inverse.get(int(code), "unseen")


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
@dataclass
class Splits:
    """A fixed, stratified train/validation/test partition shared by all models."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train_bin: np.ndarray
    y_val_bin: np.ndarray
    y_test_bin: np.ndarray
    y_train_multi: np.ndarray
    y_val_multi: np.ndarray
    y_test_multi: np.ndarray
    classes: list[str]
    pre: Preprocessor
    raw_test: pd.DataFrame          # untransformed test rows, for alert cards

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def profile_dir(profile: str) -> Path:
    return C.ARTIFACTS_DIR / profile


def build_splits(df: pd.DataFrame, *, keep_leakage: bool = False,
                 seed: int = C.SEED) -> Splits:
    """Stratified 60/20/20 split, then fit preprocessing on the training part."""
    y_multi = df[C.LABEL_MULTI].astype(str).values
    idx = np.arange(len(df))

    idx_tr, idx_tmp = train_test_split(
        idx, test_size=C.TEST_SIZE + C.VAL_SIZE, random_state=seed, stratify=y_multi)
    rel = C.TEST_SIZE / (C.TEST_SIZE + C.VAL_SIZE)
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=rel, random_state=seed, stratify=y_multi[idx_tmp])

    d_tr, d_val, d_te = df.iloc[idx_tr], df.iloc[idx_val], df.iloc[idx_te]
    pre = Preprocessor.fit(d_tr, keep_leakage=keep_leakage)

    def yb(d: pd.DataFrame) -> np.ndarray:
        return (d[C.LABEL_BINARY].astype(str) != C.BENIGN_LABEL).astype(np.int8).values

    return Splits(
        X_train=pre.transform(d_tr),
        X_val=pre.transform(d_val),
        X_test=pre.transform(d_te),
        y_train_bin=yb(d_tr), y_val_bin=yb(d_val), y_test_bin=yb(d_te),
        y_train_multi=d_tr[C.LABEL_MULTI].astype(str).values,
        y_val_multi=d_val[C.LABEL_MULTI].astype(str).values,
        y_test_multi=d_te[C.LABEL_MULTI].astype(str).values,
        classes=sorted(pd.unique(df[C.LABEL_MULTI].astype(str)).tolist()),
        pre=pre,
        raw_test=d_te.reset_index(drop=True),
    )


def save_splits(s: Splits, profile: str) -> Path:
    out = profile_dir(profile)
    out.mkdir(parents=True, exist_ok=True)
    for name, X, yb, ym in (
        ("train", s.X_train, s.y_train_bin, s.y_train_multi),
        ("val", s.X_val, s.y_val_bin, s.y_val_multi),
        ("test", s.X_test, s.y_test_bin, s.y_test_multi),
    ):
        frame = X.reset_index(drop=True).copy()
        frame["__y_bin"] = yb
        frame["__y_multi"] = ym
        frame.to_parquet(out / f"{name}.parquet", index=False)
    s.raw_test.to_parquet(out / "test_raw.parquet", index=False)
    (out / "preprocessor.json").write_text(
        json.dumps({"preprocessor": s.pre.to_json(), "classes": s.classes},
                   indent=2), encoding="utf-8")
    return out


def load_splits(profile: str) -> Splits:
    out = profile_dir(profile)
    meta_path = out / "preprocessor.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Prepared data for profile '{profile}' not found.\n"
            f"Run:  python run.py prepare --profile {profile}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pre = Preprocessor.from_json(meta["preprocessor"])

    def rd(name: str):
        f = pd.read_parquet(out / f"{name}.parquet")
        yb = f.pop("__y_bin").values.astype(np.int8)
        ym = f.pop("__y_multi").values.astype(str)
        return f, yb, ym

    Xtr, ybtr, ymtr = rd("train")
    Xv, ybv, ymv = rd("val")
    Xte, ybte, ymte = rd("test")
    return Splits(
        X_train=Xtr, X_val=Xv, X_test=Xte,
        y_train_bin=ybtr, y_val_bin=ybv, y_test_bin=ybte,
        y_train_multi=ymtr, y_val_multi=ymv, y_test_multi=ymte,
        classes=meta["classes"], pre=pre,
        raw_test=pd.read_parquet(out / "test_raw.parquet"),
    )


# --------------------------------------------------------------------------
# Small helpers shared by several stages
# --------------------------------------------------------------------------
def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default),
                    encoding="utf-8")


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
