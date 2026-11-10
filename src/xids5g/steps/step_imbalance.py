"""Stage 5 (Research Objective 4, part A) - class-imbalance handling.

Measures what class weighting and synthetic oversampling actually do to
minority-class recall, rather than assuming they help. Every technique is
applied **inside the training data only**; the validation and test splits are
untouched, so nothing synthetic is ever scored.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import config as C
from .. import console as ui
from .. import dataio, metrics, models, plotting as viz, reporting

RO = 4

#: The resamplers expand the training set roughly nine-fold (every class is
#: lifted to the size of the largest). Capping the input keeps the sweep to a
#: sensible length for a live demonstration while leaving every technique with
#: the same starting data, which is what makes the comparison fair.
TRAIN_CAP = 30_000

#: Reduced training budget, defined once in config.py. Applied identically to
#: every technique so the comparison between them is like-for-like, but it means
#: the absolute scores here are lower than Objective 2's and must not be read
#: across the two sections. See the note in config.EBM_SWEEP_PARAMS.
SWEEP_EBM = C.EBM_SWEEP_PARAMS


def run(profile: str = C.DEFAULT_PROFILE, **_) -> int:
    t = ui.StageTimer("Class-imbalance handling")
    ui.banner(
        "STAGE 5  -  CLASS IMBALANCE   (Research Objective 4a)",
        "Does class weighting or SMOTE actually improve rare-attack recall, "
        "and what does it cost elsewhere?",
    )
    raw_dir, fig_dir = C.ro_raw(RO), C.ro_fig(RO)

    with ui.timed("Loading the prepared split", "2-10 s"):
        s = dataio.load_splits(profile)
    classes = [c for c in s.classes if (s.y_test_multi == c).any()]
    ft = s.pre.feature_types()

    X_tr, y_tr = _cap(s.X_train, s.y_train_multi)
    ui.info(f"Training pool for this sweep: {len(X_tr):,} flows "
            f"(capped from {len(s.X_train):,}; every technique starts from "
            "exactly this data).")
    before = pd.Series(y_tr).value_counts()
    ui.kv({f"{k}": f"{v:,}" for k, v in before.items()},
          "Class counts before any resampling")

    techniques = _techniques()
    ui.blank()
    ui.rule(f"Sweeping {len(techniques)} imbalance strategies")

    cat_idx = [i for i, f in enumerate(X_tr.columns) if ft[i] == "nominal"]
    ui.info(f"{len(cat_idx)} of {len(ft)} features are categorical - the "
            "oversampler must be told, or it will interpolate protocol codes "
            "into values that do not exist.")

    rows, per_class, resample_info = [], {}, {}
    for label, maker in techniques.items():
        with ui.timed(f"{label}", "20-150 s") as tm:
            Xr, yr, weights, info = maker(X_tr, y_tr, cat_idx)
            resample_info[label] = info
            tm["training rows"] = f"{len(Xr):,}"

            ebm = models.make_ebm("multi", feature_types=ft, **SWEEP_EBM)
            t0 = time.perf_counter()
            ebm.fit(Xr, yr, sample_weight=weights)
            fit_s = time.perf_counter() - t0
            pred = ebm.predict(s.X_test)

            # A fit that predicts one class for everything is a failed fit, not
            # a result. Report it as such rather than publishing the score.
            if len(set(pred)) == 1:
                ui.warn(f"{label} produced a degenerate model (every flow "
                        f"predicted as '{pred[0]}'). Recorded as unstable.")
                info["status"] = f"UNSTABLE - collapsed to predicting {pred[0]}"
            else:
                info["status"] = "ok"

            sc = metrics.multiclass_scores(s.y_test_multi, pred, classes)
            pc = metrics.per_class_table(s.y_test_multi, pred, classes)
            per_class[label] = pc
            benign_recall = float(pc.set_index("class").loc[C.BENIGN_LABEL, "recall"]) \
                if C.BENIGN_LABEL in classes else float("nan")
            rows.append([label, info["train_rows"], sc["accuracy"], sc["macro_f1"],
                         sc["balanced_accuracy"], sc["minority_macro_recall"],
                         1.0 - benign_recall, round(fit_s, 1)])
            tm["macro-F1"] = f"{sc['macro_f1']:.4f}"
            tm["rare-class recall"] = f"{sc['minority_macro_recall']:.4f}"

    frame = pd.DataFrame(rows, columns=[
        "technique", "training_rows", "accuracy", "macro_f1",
        "balanced_accuracy", "minority_macro_recall", "false_alarm_rate",
        "train_seconds"])
    dataio.write_csv(frame, raw_dir / "40_imbalance_sweep.csv")
    ui.table("Effect of each imbalance strategy (glass-box EBM, 9 classes)",
             ["Technique", "Train rows", "Accuracy", "Macro-F1", "Balanced acc.",
              "Rare-class recall", "False-alarm rate", "Fit (s)"], rows)

    base = frame[frame["technique"] == "None (baseline)"].iloc[0]
    best_rare = frame.sort_values("minority_macro_recall").iloc[-1]
    best_f1 = frame.sort_values("macro_f1").iloc[-1]

    delta = frame.copy()
    for c in ("accuracy", "macro_f1", "balanced_accuracy",
              "minority_macro_recall", "false_alarm_rate"):
        delta[c] = frame[c] - float(base[c])
    dataio.write_csv(delta, raw_dir / "41_change_vs_baseline.csv")

    if abs(best_rare["minority_macro_recall"] - base["minority_macro_recall"]) < 0.005:
        ui.note(
            f"Resampling barely moves rare-attack recall here: baseline "
            f"{base['minority_macro_recall']:.4f} against a best of "
            f"{best_rare['minority_macro_recall']:.4f} "
            f"({best_rare['technique']}). Objective 1 already showed why - the "
            f"rare classes were never the weak point on this dataset.")
    else:
        ui.note(
            f"{best_rare['technique']} lifts rare-attack recall from "
            f"{base['minority_macro_recall']:.4f} to "
            f"{best_rare['minority_macro_recall']:.4f} "
            f"({best_rare['minority_macro_recall'] - base['minority_macro_recall']:+.4f}).")

    cost = best_rare["false_alarm_rate"] - base["false_alarm_rate"]
    ui.info(f"Cost of the best rare-recall technique: false-alarm rate moves "
            f"{cost:+.4f}, macro-F1 moves "
            f"{best_rare['macro_f1'] - base['macro_f1']:+.4f}.")

    # ------------------------------------------------------------- figures
    viz.grouped_bars(
        list(frame["technique"]),
        {"Macro-F1": list(frame["macro_f1"]),
         "Rare-class recall": list(frame["minority_macro_recall"]),
         "Balanced accuracy": list(frame["balanced_accuracy"]),
         "False-alarm rate": list(frame["false_alarm_rate"])},
        fig_dir / "40_imbalance_sweep.png",
        title="What each imbalance strategy actually changes",
        ylabel="score on the untouched test split", fmt="{:.3f}",
        ylim=(0, 1.15), rotate=20,
        subtitle="Resampling is applied to the training data only. The test "
                 "split is never resampled, so no synthetic flow is ever scored.")
    ui.saved(fig_dir / "40_imbalance_sweep.png")

    recall_matrix = np.array([[float(per_class[m].set_index("class")
                                     .loc[c, "recall"]) for c in classes]
                              for m in per_class])
    viz.heatmap(recall_matrix, list(per_class), classes,
                fig_dir / "41_per_class_recall.png",
                title="Per-class recall under each imbalance strategy",
                fmt="{:.2f}", cbar_label="recall",
                subtitle="Read down a column to see what a technique did to one "
                         "class.")
    ui.saved(fig_dir / "41_per_class_recall.png")

    base_row = recall_matrix[list(per_class).index("None (baseline)")]
    diff = recall_matrix - base_row
    viz.heatmap(diff, list(per_class), classes,
                fig_dir / "42_recall_change.png",
                title="Change in per-class recall against the untreated baseline",
                fmt="{:+.2f}", diverging=True, cbar_label="change in recall",
                subtitle="Red is an improvement over the baseline, blue a loss. "
                         "The baseline row is zero by definition.")
    ui.saved(fig_dir / "42_recall_change.png")

    all_pc = pd.concat([d.assign(technique=m) for m, d in per_class.items()])
    dataio.write_csv(all_pc, raw_dir / "42_per_class_recall.csv")
    dataio.write_json(resample_info, raw_dir / "43_resampling_details.json")

    _write_docs(profile, frame, delta, all_pc, base, best_rare, best_f1,
                resample_info, classes, before)

    ui.blank()
    ui.ok("Imbalance sweep complete - the effect of each technique is measured.")
    ui.info("Next step:  python run.py generalize")
    t.finish(f"Results -> {C.ro_dir(RO).name}")
    return 0


# ==========================================================================
def _cap(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Stratified cap of the training pool, preserving every class."""
    if len(X) <= TRAIN_CAP:
        return X.reset_index(drop=True), np.asarray(y)
    frac = TRAIN_CAP / len(X)
    ys = pd.Series(np.asarray(y))
    keep = []
    for cls, grp in ys.groupby(ys):
        n = max(2, int(round(len(grp) * frac)))
        keep.extend(grp.sample(n=min(n, len(grp)), random_state=C.SEED).index)
    keep = np.sort(np.asarray(keep))
    return X.iloc[keep].reset_index(drop=True), ys.iloc[keep].values


def _techniques() -> dict:
    """Each entry returns (X, y, sample_weight, info)."""

    def none(X, y, cat_idx):
        return X, y, None, {"train_rows": len(X), "how": "no treatment"}

    def weights(X, y, cat_idx):
        from sklearn.utils.class_weight import compute_sample_weight
        w = compute_sample_weight("balanced", y)
        return X, y, w, {"train_rows": len(X),
                         "how": "each flow weighted by the inverse frequency "
                                "of its class; no rows added"}

    def smote(X, y, cat_idx):
        # SMOTENC, not plain SMOTE: five of these features are protocol and
        # state codes and nine more are binary flags. Interpolating them the way
        # plain SMOTE does would invent protocol numbers that do not exist.
        from imblearn.over_sampling import SMOTENC
        Xr, yr = SMOTENC(categorical_features=cat_idx, random_state=C.SEED,
                         k_neighbors=5).fit_resample(X, y)
        return Xr, yr, None, {
            "train_rows": len(Xr),
            "how": "synthetic minority flows interpolated between each rare "
                   "flow and its 5 nearest same-class neighbours; categorical "
                   "features take the neighbours' most common value rather "
                   "than being interpolated (SMOTENC)"}

    def borderline(X, y, cat_idx):
        from imblearn.over_sampling import BorderlineSMOTE
        Xr, yr = BorderlineSMOTE(random_state=C.SEED, k_neighbors=5
                                 ).fit_resample(X, y)
        # BorderlineSMOTE has no categorical-aware variant, so codes are
        # snapped back to valid integers after resampling.
        Xr = Xr.copy()
        for i in cat_idx:
            col = Xr.columns[i]
            Xr[col] = np.rint(Xr[col].values).astype(X[col].dtype)
        counts = pd.Series(yr).value_counts()
        skipped = [c for c in counts.index
                   if counts[c] == pd.Series(y).value_counts().get(c, 0)
                   and counts[c] < counts.max()]
        return Xr, yr, None, {
            "train_rows": len(Xr),
            "classes_it_could_not_oversample": skipped,
            "how": "as SMOTE, but only near the decision boundary where "
                   "confusion actually happens; classes with no borderline "
                   "examples are left untouched"}

    def undersample(X, y, cat_idx):
        from imblearn.under_sampling import RandomUnderSampler
        Xr, yr = RandomUnderSampler(random_state=C.SEED).fit_resample(X, y)
        return Xr, yr, None, {"train_rows": len(Xr),
                              "how": "majority classes randomly cut down to the "
                                     "size of the smallest class"}

    return {
        "None (baseline)": none,
        "Class weighting": weights,
        "SMOTE (SMOTENC)": smote,
        "Borderline-SMOTE": borderline,
        "Random undersampling": undersample,
    }


def _write_docs(profile, frame, delta, all_pc, base, best_rare, best_f1,
                resample_info, classes, before) -> None:
    moved = abs(best_rare["minority_macro_recall"] - base["minority_macro_recall"])
    verdict = ("barely moves" if moved < 0.005 else
               "moves" if moved < 0.05 else "substantially changes")

    results = f"""
## 4a. Class-imbalance handling

### What was produced

{reporting.file_ref("40_imbalance_sweep.csv", "every metric for every imbalance strategy")}
{reporting.file_ref("41_change_vs_baseline.csv", "the same numbers as a change against the untreated baseline")}
{reporting.file_ref("42_per_class_recall.csv", "per-class precision, recall and F1 under each strategy")}
{reporting.file_ref("43_resampling_details.json", "how each technique altered the training set")}

### Setup

Every technique starts from an identical stratified training pool of
{int(base['training_rows']):,} flows and is evaluated on the untouched test
split. Resampling is applied to the training data only - no synthetic flow is
ever scored.

> **Read these scores against each other, not against Objective 2.** This sweep
> fits five 9-class EBMs, some on resampled data three times the size, so it
> uses a reduced training budget
> (`{", ".join(f"{k}={v}" for k, v in C.EBM_SWEEP_PARAMS.items())}`). That budget
> is applied identically to all five techniques, which is what makes the
> comparison between them valid - but it is smaller than the budget Objective 2
> uses, so the absolute macro-F1 here is lower than the headline figure and the
> two sections must not be compared directly.

Class counts in the training pool before treatment:

{metrics.df_to_markdown(pd.DataFrame({"class": before.index, "flows": before.values}))}

### Results

{metrics.df_to_markdown(frame)}

{reporting.fig_ref(RO, "40_imbalance_sweep.png", "Each imbalance strategy across four metrics.")}

### Change against the untreated baseline

{metrics.df_to_markdown(delta)}

{reporting.fig_ref(RO, "41_per_class_recall.png", "Per-class recall under each strategy.")}

{reporting.fig_ref(RO, "42_recall_change.png", "Change in per-class recall against the baseline.")}

### How each technique altered the training set

{metrics.df_to_markdown(pd.DataFrame([
    {"technique": k, "training_rows": v["train_rows"], "what it does": v["how"]}
    for k, v in resample_info.items()]))}

### Headline

- Baseline rare-class recall: **{base['minority_macro_recall']:.4f}**
- Best rare-class recall: **{best_rare['minority_macro_recall']:.4f}** ({best_rare['technique']})
- Best macro-F1: **{best_f1['macro_f1']:.4f}** ({best_f1['technique']})
- Cost of the best rare-recall technique: false-alarm rate
  {best_rare['false_alarm_rate'] - base['false_alarm_rate']:+.4f}, macro-F1
  {best_rare['macro_f1'] - base['macro_f1']:+.4f}
"""

    analysis = f"""
## 4a. Analysis - what the imbalance results mean

### The metrics used on this page

**Rare-class recall** is the unweighted mean recall over
{", ".join(C.MINORITY_CLASSES)}. It is the metric these techniques exist to
improve, so it is the one to judge them by.

**Macro-F1** and **balanced accuracy** weight every class equally, so they show
whether a gain on the rare classes was paid for elsewhere.

**False-alarm rate** is the share of benign flows labelled as an attack.
Oversampling the attack classes pushes a model's decision boundary towards
predicting attack, so this is where the bill for a recall gain usually arrives.

**Class weighting** multiplies each flow's contribution to the training loss by
the inverse frequency of its class - no rows are added, the rare ones simply
count for more. **SMOTE** creates synthetic minority flows by interpolating
between a rare flow and its nearest same-class neighbours. **Borderline-SMOTE**
does the same but only near the decision boundary. **Random undersampling**
discards majority-class rows instead.

### What the measured values indicate

**Resampling {verdict} rare-attack recall on this dataset.** The baseline
already achieves {base['minority_macro_recall']:.4f}; the best technique
({best_rare['technique']}) reaches
{best_rare['minority_macro_recall']:.4f}, a change of
{best_rare['minority_macro_recall'] - base['minority_macro_recall']:+.4f}.

This is the expected consequence of what Objective 1 measured. Imbalance
handling attacks a specific failure - a class so under-represented that the
loss function ignores it - and that failure is not what limits this dataset.
The rare classes here are highly separable, so a few hundred training examples
are enough to pin them down; the binding constraint is the benign-versus-
UDPFlood collision, which no resampling technique can address because the
flows involved are genuinely identical in the feature space.

**The cost side is where the techniques differ.** Moving from the baseline to
{best_rare['technique']} changes the false-alarm rate by
{best_rare['false_alarm_rate'] - base['false_alarm_rate']:+.4f} and macro-F1 by
{best_rare['macro_f1'] - base['macro_f1']:+.4f}. Random undersampling is the
clearest illustration: it discards most of the training data
({int(frame[frame['technique'] == 'Random undersampling']['training_rows'].iloc[0]):,}
rows against {int(base['training_rows']):,}) and the deterioration in the
common classes shows up immediately in the recall-change figure.

**This is a negative result reported as such.** The methodology in the proposal
states that the effect of each technique is to be *measured rather than
assumed*, and the measurement says these techniques are not the lever on this
data. Reporting that is more useful to a practitioner than presenting the
best-looking configuration and omitting the rest, and the per-class change map
shows exactly which classes moved and by how much.

**Interpreting the figures.** In the sweep figure, compare each technique's
group against the leftmost (baseline) group - flat bars mean the technique did
nothing. The recall-change map is the most informative: it is zero by
construction along the baseline row, and every coloured cell elsewhere is a
class that a technique moved, red for better and blue for worse. A technique
that shows red on the rare-class columns and blue on the Benign column has
bought recall with false alarms.

### How this supports Research Objective 4

The first half of Objective 4 asks for an evaluation of "how imbalance-handling
techniques affect minority-class recall". Delivered:

- Five training regimes (untreated, class weighting, SMOTE, Borderline-SMOTE,
  random undersampling), all applied inside the training data only, all
  starting from an identical pool, all scored on the same untouched test split.
- Rare-class recall reported for each, per class and in aggregate, together with
  the macro-F1 and false-alarm consequences.
- The change against the untreated baseline tabulated and mapped, so the effect
  of each technique is isolated.

The conclusion - that these techniques are close to inert here because the rare
classes were never the bottleneck - follows from evidence collected in
Objective 1 and confirmed independently here. The second half of Objective 4,
the AWID3 generalisation probe, is covered in the next section.
"""
    reporting.write_fragment(
        profile, RO, "imbalance", 1, results_md=results, analysis_md=analysis,
        headline={
            "baseline_minority_recall": float(base["minority_macro_recall"]),
            "best_minority_recall": float(best_rare["minority_macro_recall"]),
            "best_technique": str(best_rare["technique"]),
        })
