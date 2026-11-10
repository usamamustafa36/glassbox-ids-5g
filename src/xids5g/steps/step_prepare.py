"""Stage 1 (Research Objective 1, part A) - prepare and profile 5G-NIDD.

Covers the first half of Objective 1: clean and encode the data, quantify the
class imbalance the study is built around, audit the feature set for label
leakage, and establish the accuracy ceiling that the leakage-free feature set
actually supports. The fixed train/validation/test split written here is the
one every later stage reuses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .. import console as ui
from .. import dataio, metrics, plotting as viz, reporting

RO = 1


def run(profile: str = C.DEFAULT_PROFILE, **_) -> int:
    t = ui.StageTimer("Data preparation and profiling")
    prof = C.PROFILES[profile]
    ui.banner(
        "STAGE 1  -  DATA PREPARATION AND PROFILING   (Research Objective 1a)",
        "Clean the 5G-NIDD flows, measure the class imbalance, audit the "
        "features for label leakage, and build the fixed split all models share.",
    )
    C.ensure_dirs()
    raw_dir, fig_dir = C.ro_raw(RO), C.ro_fig(RO)

    # ---------------------------------------------------------------- load
    with ui.timed(f"Loading 5G-NIDD and taking the '{profile}' sample",
                  "10-40 s") as tm:
        df = dataio.load_raw(prof.n_rows)
        tm["flows"] = f"{len(df):,}"
        tm["columns"] = df.shape[1]
    ui.info(f"Source file: {C.NIDD_CSV.name} "
            f"({C.NIDD_CSV.stat().st_size / 1e6:,.0f} MB on disk)")
    if prof.n_rows:
        ui.info(f"Stratified sample of {len(df):,} flows drawn from 1,215,890 - "
                "class proportions preserved exactly.")

    # ------------------------------------------------------------- profile
    ui.blank()
    ui.rule("Dataset profile")
    with ui.timed("Profiling labels, class balance and missing values", "5-15 s"):
        counts = df[C.LABEL_MULTI].value_counts()
        shares = 100 * counts / len(df)
        imbalance_ratio = float(counts.max() / counts.min())
        bin_counts = df[C.LABEL_BINARY].value_counts()

        class_profile = pd.DataFrame({
            "class": counts.index,
            "flows": counts.values,
            "share_%": np.round(shares.values, 4),
            "is_minority": [c in C.MINORITY_CLASSES for c in counts.index],
        })
        dataio.write_csv(class_profile, raw_dir / "01_class_distribution.csv")

        missing = (df.isna().mean() * 100).sort_values(ascending=False)
        miss_frame = pd.DataFrame({
            "column": missing[missing > 0].index,
            "missing_%": missing[missing > 0].values,
        })
        dataio.write_csv(miss_frame, raw_dir / "02_missing_values.csv")

    ui.table("Class distribution (9-class 'Attack Type' label)",
             ["Class", "Flows", "Share %", "Rare?"],
             [[r["class"], f"{r['flows']:,}", round(r["share_%"], 3),
               "yes" if r["is_minority"] else ""]
              for _, r in class_profile.iterrows()])
    ui.kv({
        "Binary label - Benign": f"{bin_counts.get('Benign', 0):,}",
        "Binary label - Malicious": f"{bin_counts.get('Malicious', 0):,}",
        "Largest class": f"{counts.index[0]} ({shares.iloc[0]:.2f}%)",
        "Smallest class": f"{counts.index[-1]} ({shares.iloc[-1]:.3f}%)",
        "Imbalance ratio (largest : smallest)": f"{imbalance_ratio:,.0f} : 1",
        "Columns with missing values": int((missing > 0).sum()),
    }, "Headline profile")
    ui.note(f"The 9-class label is imbalanced {imbalance_ratio:,.0f}-to-1. "
            "Overall accuracy is dominated by the largest class, which is "
            "exactly why this objective reports per-class recall as well.")

    viz.barh(list(class_profile["class"]), list(class_profile["flows"]),
             fig_dir / "01_class_distribution.png",
             title="5G-NIDD class distribution (log scale hidden; counts shown)",
             xlabel="number of flows", fmt="{:,.0f}",
             highlight=[c for c in class_profile["class"] if c in C.MINORITY_CLASSES],
             subtitle="Rare classes highlighted. The smallest class holds "
                      f"{counts.iloc[-1]:,} flows against {counts.iloc[0]:,} "
                      "in the largest.")
    ui.saved(fig_dir / "01_class_distribution.png")

    top_missing = miss_frame.head(18)
    if len(top_missing):
        viz.barh(list(top_missing["column"]), list(top_missing["missing_%"]),
                 fig_dir / "02_missing_values.png",
                 title="Missing values by column (training data)",
                 xlabel="% of flows with the value absent", fmt="{:.1f}%",
                 subtitle="Absence is informative here: no destination TTL means "
                          "the peer never replied. Columns above 99% are dropped; "
                          "the rest keep an explicit 'not observed' indicator.")
        ui.saved(fig_dir / "02_missing_values.png")

    # ------------------------------------------------------- leakage audit
    ui.blank()
    ui.rule("Leakage audit")
    ui.info("5G-NIDD's Combined.csv was assembled by concatenating separate "
            "benign and per-attack captures. Any column that encodes position "
            "within the capture therefore encodes the label.")

    with ui.timed("Testing the capture-order columns (Seq, Offset) on their own",
                  "10-30 s") as tm:
        leak_report = _leakage_audit(df)
        tm["Seq+Offset alone"] = f"{leak_report['leaky_only_accuracy']:.4f} accuracy"
    dataio.write_json(leak_report, raw_dir / "03_leakage_audit.json")

    ui.table("What each feature set can achieve (XGBoost, held-out 30%)",
             ["Feature set", "Accuracy", "Verdict"],
             [["All 48 columns (Seq and Offset included)",
               leak_report["with_leakage_accuracy"],
               "inflated - not a real detection result"],
              ["Seq and Offset only (no traffic features at all)",
               leak_report["leaky_only_accuracy"],
               "proves these two columns carry the label"],
              ["Traffic features only (Seq and Offset removed)",
               leak_report["clean_accuracy"],
               "the honest detection result"]])
    ui.note(
        f"Two columns that describe nothing about the traffic - the record "
        f"sequence number and the byte offset in the capture file - reach "
        f"{leak_report['leaky_only_accuracy']:.1%} accuracy by themselves. They are "
        f"dropped from every model in this project.")

    viz.barh(["All columns\n(Seq + Offset included)", "Seq + Offset only",
              "Traffic features only\n(used by this project)"],
             [leak_report["with_leakage_accuracy"],
              leak_report["leaky_only_accuracy"],
              leak_report["clean_accuracy"]],
             fig_dir / "03_leakage_audit.png",
             title="Leakage audit: accuracy by feature set",
             xlabel="binary accuracy on held-out data", fmt="{:.4f}",
             highlight=["All columns\n(Seq + Offset included)", "Seq + Offset only"],
             subtitle="A model given only the capture-file position - and no "
                      "traffic information whatsoever - already scores very "
                      "highly. That score measures file layout, not detection.")
    ui.saved(fig_dir / "03_leakage_audit.png")

    # --------------------------------------------------------------- split
    ui.blank()
    ui.rule("Preprocessing and the fixed split")
    with ui.timed("Cleaning, encoding and splitting 60 / 20 / 20 (stratified)",
                  "10-30 s") as tm:
        splits = dataio.build_splits(df)
        out = dataio.save_splits(splits, profile)
        tm["features"] = len(splits.feature_names)
        tm["train/val/test"] = (f"{len(splits.X_train):,}/"
                                f"{len(splits.X_val):,}/{len(splits.X_test):,}")
    ui.saved(out)

    pre = splits.pre
    ui.kv({
        "Input features kept": len(pre.feature_names),
        "  of which numeric": len(pre.medians),
        "  of which categorical": len(pre.categorical_features),
        "  of which 'not observed' flags": len(pre.indicator_columns),
        "Columns removed": len(pre.drop_reasons),
        "Training flows": f"{len(splits.X_train):,}",
        "Validation flows": f"{len(splits.X_val):,}",
        "Test flows": f"{len(splits.X_test):,}",
    }, "Prepared feature matrix")

    drop_frame = pd.DataFrame(
        {"column": list(pre.drop_reasons), "reason": list(pre.drop_reasons.values())}
    ).sort_values("column")
    dataio.write_csv(drop_frame, raw_dir / "04_removed_columns.csv")
    ui.table("Columns removed before modelling",
             ["Column", "Reason"], drop_frame.values.tolist())

    dataio.write_csv(pd.DataFrame({"feature": pre.feature_names,
                                   "type": pre.feature_types()}),
                     raw_dir / "05_feature_list.csv")

    # ------------------------------------------------------------- ceiling
    ui.blank()
    ui.rule("Attainable accuracy ceiling")
    ui.info("Some flows have byte-for-byte identical feature vectors but "
            "different labels. No classifier can separate those, so they set a "
            "hard upper bound on any accuracy reported from these features.")
    with ui.timed("Computing the ceiling on the prepared features", "10-40 s") as tm:
        X_all = pd.concat([splits.X_train, splits.X_val, splits.X_test])
        y_all_b = np.concatenate([splits.y_train_bin, splits.y_val_bin,
                                  splits.y_test_bin])
        y_all_m = np.concatenate([splits.y_train_multi, splits.y_val_multi,
                                  splits.y_test_multi])
        # Hash every row once and reuse it for all three passes.
        keys = metrics.row_keys(X_all)
        ceil_bin = metrics.bayes_ceiling(X_all, y_all_b, key=keys)
        ceil_multi = metrics.bayes_ceiling(X_all, y_all_m, key=keys)
        tm["binary ceiling"] = f"{ceil_bin['ceiling_accuracy']:.4f}"
        tm["9-class ceiling"] = f"{ceil_multi['ceiling_accuracy']:.4f}"

    collisions = _collision_breakdown(keys, y_all_m)
    dataio.write_json({"binary": ceil_bin, "multiclass": ceil_multi,
                       "classes_in_conflicting_groups": collisions},
                      raw_dir / "06_accuracy_ceiling.json")

    ui.kv({
        "Best accuracy any model can reach (binary)": ceil_bin["ceiling_accuracy"],
        "Best accuracy any model can reach (9-class)": ceil_multi["ceiling_accuracy"],
        "Irreducible error": ceil_bin["irreducible_error"],
        "Distinct feature vectors": f"{ceil_bin['n_distinct_feature_vectors']:,}",
        "Flows sharing a vector with a different label":
            f"{ceil_bin['n_rows_in_conflicting_groups']:,} "
            f"({ceil_bin['pct_rows_in_conflicting_groups']:.1f}%)",
    }, "Ceiling on the leakage-free feature set")
    ui.note(
        f"{ceil_bin['pct_rows_in_conflicting_groups']:.0f}% of flows are "
        f"indistinguishable from a flow with a different label. The ceiling is "
        f"{ceil_bin['ceiling_accuracy']:.2%}, so every accuracy in this project "
        f"must be read against that number - not against 100%.")

    if collisions:
        top = sorted(collisions.items(), key=lambda kv: -kv[1])[:6]
        viz.barh([k for k, _ in top], [v for _, v in top],
                 fig_dir / "04_collision_classes.png",
                 title="Which classes are indistinguishable from one another",
                 xlabel="flows sitting in a group with conflicting labels",
                 fmt="{:,.0f}",
                 subtitle="These flows share an identical feature vector with a "
                          "flow of another class. They set the accuracy ceiling.")
        ui.saved(fig_dir / "04_collision_classes.png")

    viz.stat_tiles(
        [("Flows analysed", f"{len(df):,}", f"{profile} profile"),
         ("Class imbalance", f"{imbalance_ratio:,.0f}:1", "largest vs smallest class"),
         ("Accuracy ceiling", f"{ceil_bin['ceiling_accuracy']:.1%}",
          "best any model can do without leakage"),
         ("Leaked accuracy", f"{leak_report['leaky_only_accuracy']:.1%}",
          "from capture position alone")],
        fig_dir / "00_headline.png",
        title="Research Objective 1a - headline numbers")
    ui.saved(fig_dir / "00_headline.png")

    _write_docs(profile, df, class_profile, imbalance_ratio, bin_counts,
                leak_report, ceil_bin, ceil_multi, collisions, splits, miss_frame)

    ui.blank()
    ui.ok("Prepared data cached. Every later stage reuses this exact split.")
    ui.info("Next step:  python run.py baselines")
    t.finish(f"Results -> {C.ro_dir(RO).name}")
    return 0


# ==========================================================================
# helpers
# ==========================================================================
def _leakage_audit(df: pd.DataFrame) -> dict:
    """Quantify how much of the achievable accuracy comes from capture order."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    n = min(len(df), 150_000)
    sample = df.sample(n=n, random_state=C.SEED) if n < len(df) else df
    y = (sample[C.LABEL_BINARY].astype(str) != C.BENIGN_LABEL).astype(int).values

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].astype("category").cat.codes.replace(-1, np.nan)
        return out.astype(np.float32)

    def score(frame: pd.DataFrame) -> float:
        X = encode(frame)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=C.SEED, stratify=y)
        m = xgb.XGBClassifier(n_estimators=120, max_depth=8, tree_method="hist",
                              n_jobs=C.N_JOBS, random_state=C.SEED)
        m.fit(Xtr, ytr)
        return float(m.score(Xte, yte))

    all_cols = sample.drop(columns=C.LABEL_COLUMNS + ["Unnamed: 0"],
                           errors="ignore")
    clean_cols = all_cols.drop(columns=[c for c in C.LEAKAGE_COLUMNS
                                        if c in all_cols.columns])
    leak_cols = sample[[c for c in ("Seq", "Offset") if c in sample.columns]]

    with_leak = score(all_cols)
    leak_only = score(leak_cols)
    clean = score(clean_cols)
    return {
        "rows_used": int(n),
        "with_leakage_accuracy": with_leak,
        "leaky_only_accuracy": leak_only,
        "clean_accuracy": clean,
        "accuracy_attributable_to_leakage": with_leak - clean,
        "leaky_columns": list(leak_cols.columns),
        "note": ("Seq is the Argus record sequence number and Offset the byte "
                 "offset within the capture file. Because Combined.csv is a "
                 "concatenation of separate benign and per-attack captures, "
                 "both encode the label. They are excluded from all modelling "
                 "in this project."),
    }


def _collision_breakdown(key: np.ndarray, y: np.ndarray) -> dict[str, int]:
    """How many flows of each class sit in a label-conflicting group."""
    frame = pd.DataFrame({"k": key, "y": np.asarray(y).astype(str)})
    conflicting = frame.groupby("k")["y"].transform("nunique") > 1
    return frame.loc[conflicting, "y"].value_counts().to_dict()


def _write_docs(profile, df, class_profile, imbalance_ratio, bin_counts,
                leak, ceil_bin, ceil_multi, collisions, splits, miss_frame) -> None:
    pre = splits.pre
    top_coll = sorted(collisions.items(), key=lambda kv: -kv[1])[:4]
    coll_txt = ", ".join(f"{k} ({v:,} flows)" for k, v in top_coll) or "none"

    results = f"""
## 1a. Dataset preparation and profiling

### What was produced

{reporting.file_ref("01_class_distribution.csv", "flows and percentage share of each of the 9 attack classes")}
{reporting.file_ref("02_missing_values.csv", "percentage of missing values per column")}
{reporting.file_ref("03_leakage_audit.json", "accuracy reachable from each feature set, with and without the capture-order columns")}
{reporting.file_ref("04_removed_columns.csv", "every column removed before modelling and the reason for it")}
{reporting.file_ref("05_feature_list.csv", "the final feature list with each feature's type")}
{reporting.file_ref("06_accuracy_ceiling.json", "the highest accuracy any model can reach on these features")}

### Class distribution

The dataset carries a binary label (`Label`) and a 9-class label (`Attack Type`).

{reporting.kv_table({
    "Flows analysed": len(df),
    "Benign flows": int(bin_counts.get("Benign", 0)),
    "Malicious flows": int(bin_counts.get("Malicious", 0)),
    "Attack classes": int(len(class_profile) - 1),
    "Largest class": f"{class_profile.iloc[0]['class']} "
                     f"({class_profile.iloc[0]['share_%']:.2f}%)",
    "Smallest class": f"{class_profile.iloc[-1]['class']} "
                      f"({class_profile.iloc[-1]['share_%']:.3f}%)",
    "Imbalance ratio": f"{imbalance_ratio:,.0f} : 1",
}, "Quantity", "Measured value")}

{metrics.df_to_markdown(class_profile.assign(
    flows=class_profile["flows"].map("{:,}".format)), "{:.3f}")}

{reporting.fig_ref(RO, "01_class_distribution.png",
                   "Flows per class. The five rare classes are highlighted.")}

The rare classes - {", ".join(C.MINORITY_CLASSES)} - together account for a
small fraction of the data but are the reconnaissance and low-volume signals
that matter operationally. An overall accuracy figure can be high while every
one of them is missed, which is why this objective reports per-class recall
separately.

### Missing values

{len(miss_frame)} of the 52 columns contain missing values.

{reporting.fig_ref(RO, "02_missing_values.png", "Missing-value rate by column.")}

Missingness here is informative rather than accidental: a flow with no
destination TTL is one the peer never answered. Two columns (`sVid`, `dVid`)
are missing in over 99% of flows and are dropped; for the remainder the value
is filled with the training-set median **and** an explicit
`<column>__not_observed` flag is added, so the model can use the fact that the
value was absent.

### Leakage audit

{reporting.kv_table({
    "Accuracy using all 48 columns": leak["with_leakage_accuracy"],
    "Accuracy using ONLY Seq and Offset": leak["leaky_only_accuracy"],
    "Accuracy using traffic features only": leak["clean_accuracy"],
    "Accuracy attributable to leakage": leak["accuracy_attributable_to_leakage"],
}, "Feature set", "Binary accuracy")}

{reporting.fig_ref(RO, "03_leakage_audit.png",
                   "Accuracy reachable from each feature set.")}

`Seq` (the Argus record sequence number) and `Offset` (the byte offset inside
the capture file) describe *where a record sits in the file*, not what the
traffic did. `Combined.csv` was built by concatenating separate benign and
per-attack captures, so position within the file is a near-perfect proxy for
the label. Both columns are removed from every model in this project.

### Attainable accuracy ceiling

{reporting.kv_table({
    "Highest reachable binary accuracy": ceil_bin["ceiling_accuracy"],
    "Highest reachable 9-class accuracy": ceil_multi["ceiling_accuracy"],
    "Irreducible error": ceil_bin["irreducible_error"],
    "Distinct feature vectors": ceil_bin["n_distinct_feature_vectors"],
    "Flows in label-conflicting groups": ceil_bin["n_rows_in_conflicting_groups"],
    "Share of flows in conflicting groups":
        f"{ceil_bin['pct_rows_in_conflicting_groups']:.2f}%",
}, "Quantity", "Measured value")}

Classes most affected: {coll_txt}.

{reporting.fig_ref(RO, "04_collision_classes.png",
                   "Flows sitting in identical-feature groups with conflicting labels.")}

### Prepared split

{reporting.kv_table({
    "Features used": len(pre.feature_names),
    "Numeric features": len(pre.medians),
    "Categorical features": len(pre.categorical_features),
    "'Not observed' indicator features": len(pre.indicator_columns),
    "Training flows": len(splits.X_train),
    "Validation flows": len(splits.X_val),
    "Test flows": len(splits.X_test),
    "Split": "60 / 20 / 20, stratified on the 9-class label",
    "Random seed": C.SEED,
}, "Quantity", "Value")}
"""

    analysis = f"""
## 1a. Analysis - what the preparation numbers mean

### The metrics used on this page

**Class share (%)** is the fraction of all flows belonging to one class. It
matters because a classifier that always predicts the largest class already
scores that share as its accuracy - here
{class_profile.iloc[0]['share_%']:.1f}% - without learning anything.

**Imbalance ratio** is the size of the largest class divided by the size of the
smallest. Measured value: **{imbalance_ratio:,.0f} : 1**. For every
{class_profile.iloc[-1]['class']} flow in the data there are roughly
{imbalance_ratio:,.0f} {class_profile.iloc[0]['class']} flows. A model trained
to minimise average error is therefore rewarded for ignoring
{class_profile.iloc[-1]['class']} entirely, which is the specific failure
Objective 1 sets out to expose and Objective 4 sets out to fix.

**Missing rate** is the fraction of flows for which a column has no value. In
network-flow data this is a signal, not noise - the absence of a reverse-
direction statistic means there was no reverse-direction traffic.

**Accuracy ceiling (Bayes-optimal accuracy)** is the best score obtainable by
*any* classifier on a given feature representation. It is computed exactly
here: flows with byte-for-byte identical feature vectors cannot be separated by
any function of those features, so the optimum is to predict the majority label
inside each identical-feature group. The accuracy of that rule is the ceiling.

### What the measured values indicate

**The imbalance is severe and real.** At {imbalance_ratio:,.0f}:1 the dataset
sits well outside the range where plain accuracy is meaningful. This confirms
research gap 3 from the proposal quantitatively, on the data itself, and it
justifies the per-class recall reporting used throughout the remaining
objectives.

**Two columns in the published CSV leak the label.** This is the most
consequential finding of the preparation stage. `Seq` and `Offset` are
artefacts of how the dataset file was assembled. On their own, with no
information about packets, bytes, protocols or timing, they reach
**{leak["leaky_only_accuracy"]:.2%}** binary accuracy. Including them lifts a
standard XGBoost model from **{leak["clean_accuracy"]:.2%}** to
**{leak["with_leakage_accuracy"]:.2%}** - a gap of
{leak["accuracy_attributable_to_leakage"]:.1%} that measures file layout rather
than intrusion detection. Any published figure near 99.9% on this CSV needs to
state whether these columns were excluded; where a study does not say, its
headline number is not directly comparable with the ones reported here. This
project removes them, so every result that follows is a genuine
traffic-based detection result.

**The honest ceiling is {ceil_bin["ceiling_accuracy"]:.2%}, not 100%.**
{ceil_bin["pct_rows_in_conflicting_groups"]:.1f}% of flows share an identical
feature vector with a flow carrying a different label. The dominant collision
is between benign traffic and {top_coll[0][0] if top_coll else "the flooding classes"}:
a single-packet benign UDP exchange and a single-packet UDP flood packet
produce the same Argus flow record. No model, glass-box or black-box, can
separate them from these features. This reframes every accuracy number in the
project: a detector scoring {ceil_bin["ceiling_accuracy"] - 0.005:.2%} is not a
mediocre detector, it is a detector operating at the limit of what the data
allows.

**Interpreting the figures.** In the class-distribution figure the bar lengths
span three orders of magnitude - the visual gap between the top and bottom bars
*is* the imbalance problem. In the leakage-audit figure, the middle bar is the
one to read: it uses no traffic information at all, yet nearly matches the full
model. In the collision figure, the two tall bars identify precisely which
classes the ceiling comes from.

### How this supports Research Objective 1

Objective 1 asks for the dataset to be prepared and profiled, with the
minority-class picture made explicit. This stage delivers that and adds two
pieces of due diligence the objective implies but does not spell out:

1. The data is cleaned, encoded and split once, into a fixed stratified
   60/20/20 partition that every subsequent model reuses. This is what makes
   the glass-box versus black-box comparison in Objectives 2 and 3 a
   like-for-like comparison rather than an approximate one.
2. The feature set is audited before any model is trusted. Removing `Seq` and
   `Offset` and establishing the {ceil_bin["ceiling_accuracy"]:.2%} ceiling
   means the accuracy comparisons that follow are measuring detection skill,
   and are being judged against an attainable target.

The remaining half of Objective 1 - baseline detectors and their rare-class
detection rates - is covered in the next section.
"""
    reporting.write_fragment(
        profile, RO, "prepare", 1, results_md=results, analysis_md=analysis,
        headline={
            "imbalance_ratio": imbalance_ratio,
            "ceiling_binary": ceil_bin["ceiling_accuracy"],
            "ceiling_multiclass": ceil_multi["ceiling_accuracy"],
            "leaky_only_accuracy": leak["leaky_only_accuracy"],
            "n_features": len(pre.feature_names),
        })
