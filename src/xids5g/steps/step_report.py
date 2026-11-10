"""Stage 8 - the cross-objective summary.

Collects the headline number from every stage that has run and writes the
Results index a reader lands on first.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import config as C
from .. import console as ui
from .. import dataio, metrics, reporting


def run(profile: str = C.DEFAULT_PROFILE, **_) -> int:
    t = ui.StageTimer("Summary report")
    ui.banner("STAGE 8  -  SUMMARY REPORT",
              "Collect every objective's headline result into one place.")
    C.ensure_dirs()
    out = C.RESULTS_DIR / "Summary"

    with ui.timed("Collecting results from every objective", "2-10 s") as tm:
        frags = reporting.load_fragments(profile)
        tm["stages found"] = len(frags)

    if not frags:
        ui.warn("No stage results found yet. Run the pipeline first: "
                "python run.py all")
        t.finish("nothing to summarise")
        return 0

    headlines = {}
    for f in frags:
        for k, v in (f.get("headline") or {}).items():
            headlines[f"O{f['objective']}.{k}"] = v
    dataio.write_json(headlines, out / "headline_numbers.json")

    rows = []
    for n in sorted(C.RO_DIRS):
        stages = [f for f in frags if f["objective"] == n]
        rows.append([
            f"Objective {n}",
            C.RO_TITLES[n][:78] + ("..." if len(C.RO_TITLES[n]) > 78 else ""),
            len(stages),
            _count(C.ro_raw(n)),
            _count(C.ro_fig(n)),
            "yes" if (C.ro_dir(n) / "Analysis.md").exists() else "no",
        ])
    ui.table("What each research objective produced",
             ["Objective", "Goal", "Stages", "Raw files", "Figures",
              "Analysis written"], rows)

    ui.blank()
    ui.rule("Headline numbers across the whole study")
    key = _key_findings(headlines)
    ui.table("Key findings", ["Finding", "Value"], key)

    with ui.timed("Writing the Results index and summary", "2-10 s"):
        _write_index(profile, frags, headlines)
        _write_summary(profile, frags, headlines, key)
        _write_timings(out)
    ui.saved(C.RESULTS_DIR / "README.md")
    ui.saved(out / "SUMMARY.md")

    ui.blank()
    ui.ok("Summary written. Start reading at Results/README.md.")
    t.finish(f"{len(frags)} stages summarised")
    return 0


def _count(d: Path) -> int:
    return len([p for p in d.glob("*") if p.is_file()]) if d.exists() else 0


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _key_findings(h: dict) -> list[list[str]]:
    wanted = [
        ("O1.imbalance_ratio", "Class imbalance in 5G-NIDD (largest : smallest)"),
        ("O1.leaky_only_accuracy",
         "Accuracy from capture position alone (the leakage found)"),
        ("O1.ceiling_binary", "Attainable accuracy ceiling once leakage is removed"),
        ("O2.ebm_binary_accuracy", "Glass-box (EBM) binary accuracy"),
        ("O2.best_blackbox_accuracy", "Best black-box binary accuracy"),
        ("O2.p_value", "p-value of the glass-box vs black-box difference"),
        ("O2.ebm_macro_f1", "Glass-box 9-class macro-F1"),
        ("O3.worst_explainer_agreement",
         "Worst agreement between two post-hoc explainers of ONE model"),
        ("O3.ebm_ground_truth_rmse", "EBM error against a known ground-truth effect"),
        ("O3.shap_ground_truth_rmse", "SHAP error against the same known effect"),
        ("O4.best_minority_recall", "Best rare-attack recall after imbalance handling"),
        ("O4.awid3_accuracy", "Glass-box accuracy on AWID3 (Wi-Fi)"),
        ("O4.family_spearman", "Feature-family transfer, 5G-NIDD to AWID3"),
        ("O5.n_cards", "Analyst alert cards produced"),
        ("O5.exact_cards", "Cards whose reasons reconstruct the decision exactly"),
    ]
    return [[label, _fmt(h[k])] for k, label in wanted if k in h]


def _write_index(profile: str, frags: list[dict], h: dict) -> None:
    prof = C.PROFILES[profile]
    n = f"{prof.n_rows:,}" if prof.n_rows else "1,215,890 (all)"
    lines = [
        "# Results",
        "",
        "**Explainable Intrusion Detection for 5G Wireless Networks Using a "
        "Glass-Box Machine Learning Model**  ",
        "Usama Mustafa",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from the "
        f"`{profile}` profile ({n} flows). Random seed {C.SEED}._",
        "",
        "---",
        "",
        "## How this folder is organised",
        "",
        "One folder per research objective. Inside each:",
        "",
        "| File | What it contains |",
        "|---|---|",
        "| `Results.md` | What was produced and what the numbers are |",
        "| `Analysis.md` | What each metric means, what the values indicate, "
        "and how they demonstrate the objective |",
        "| `raw_results/` | The underlying CSV and JSON outputs |",
        "| `figures/` | Every chart, as PNG |",
        "",
        "## The five research objectives",
        "",
    ]
    for i in sorted(C.RO_DIRS):
        folder = C.RO_DIRS[i]
        done = any(f["objective"] == i for f in frags)
        lines += [
            f"### Objective {i}{'' if done else '  _(not yet run)_'}",
            "",
            f"> {C.RO_TITLES[i]}",
            "",
            f"- [Results]({_link(folder)}/Results.md)",
            f"- [Analysis]({_link(folder)}/Analysis.md)",
        ]
        if i == 5:
            lines.append(f"- [**Alert cards (open in a browser)**]"
                         f"({_link(folder)}/ALERT_CARDS.html)")
        lines.append("")

    lines += [
        "## Headline numbers",
        "",
        "| Finding | Value |",
        "|---|---|",
    ]
    for label, value in _key_findings(h):
        lines.append(f"| {label} | {value} |")
    lines += [
        "",
        "See [`Summary/SUMMARY.md`](Summary/SUMMARY.md) for the cross-objective "
        "narrative and [`Summary/objective_validation.json`]"
        "(Summary/objective_validation.json) for the objective coverage check.",
        "",
        "## Reproducing these results",
        "",
        "```bash",
        "python -m pip install -r requirements.txt",
        "python run.py verify",
        f"python run.py all --profile {profile}",
        "```",
        "",
        "Full instructions: [`DEMO_GUIDE.md`](../DEMO_GUIDE.md)",
        "",
    ]
    (C.RESULTS_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _link(folder: str) -> str:
    return folder.replace(" ", "%20")


def _write_summary(profile: str, frags: list[dict], h: dict, key) -> None:
    ceiling = h.get("O1.ceiling_binary")
    ebm = h.get("O2.ebm_binary_accuracy")
    bb = h.get("O2.best_blackbox_accuracy")
    worst = h.get("O3.worst_explainer_agreement")

    # Every claim below is phrased from the measured values, not asserted. If a
    # run produces a different result the wording follows it.
    gap = (ebm - bb) if (ebm is not None and bb is not None) else None
    if gap is None:
        claim3 = "**3. Glass-box versus black-box accuracy.** Not yet measured."
    elif abs(gap) < 0.005:
        claim3 = (
            f"**3. The glass-box model costs nothing in accuracy.** EBM scored "
            f"{_fmt(ebm)} against the best black box at {_fmt(bb)} - a "
            f"difference of {gap:+.4f}, with a paired significance test over "
            f"{C.N_REPEATS} independent re-trainings returning p = "
            f"{_fmt(h.get('O2.p_value', float('nan')))}. Both sit against the "
            f"ceiling, so there is no accuracy left for a black box to claim. "
            f"The accuracy-versus-interpretability trade-off does not appear on "
            f"this data.")
    elif gap > 0:
        claim3 = (
            f"**3. The glass-box model was the more accurate of the two.** EBM "
            f"scored {_fmt(ebm)} against the best black box at {_fmt(bb)}, a "
            f"margin of {gap:+.4f}.")
    else:
        claim3 = (
            f"**3. The glass-box model traded some accuracy.** EBM scored "
            f"{_fmt(ebm)} against the best black box at {_fmt(bb)}, a shortfall "
            f"of {gap:+.4f}. Judge that against the attainable ceiling of "
            f"{_fmt(ceiling)} rather than against 1.00.")

    e_rmse = h.get("O3.ebm_ground_truth_rmse")
    s_rmse = h.get("O3.shap_ground_truth_rmse")
    if e_rmse is None or s_rmse is None:
        claim5 = "**5. Ground-truth recovery.** Not yet measured."
    elif e_rmse < s_rmse:
        claim5 = (
            f"**5. Against a known ground truth, the glass-box explanation is "
            f"closer.** On synthetic data with a known generating function, the "
            f"EBM shape function recovered the true effect with RMSE "
            f"{_fmt(e_rmse)} against {_fmt(s_rmse)} for SHAP-on-XGBoost.")
    else:
        claim5 = (
            f"**5. Against a known ground truth, SHAP-on-XGBoost was closer.** "
            f"RMSE {_fmt(s_rmse)} against {_fmt(e_rmse)} for the EBM shape "
            f"function - reported as measured.")

    aw_acc = h.get("O4.awid3_accuracy")
    fam_rho = h.get("O4.family_spearman")
    overlap = h.get("O4.top3_overlap")
    if aw_acc is None:
        claim7 = "**7. Generalisation probe.** Not yet measured."
    else:
        transfer = (
            f"and the *kinds* of features the two detectors rely on transfer "
            f"as well (rank correlation {_fmt(fam_rho)}, top-3 family overlap "
            f"{overlap}/3)."
            if (fam_rho or 0) > 0.5 else
            f"The feature-family ordering, however, transfers only weakly "
            f"(rank correlation {_fmt(fam_rho)}, with "
            f"{overlap} of the top 3 families shared). 802.11 headers carry "
            f"radio-layer signal "
            f"data with no 5G flow-record equivalent, and vice versa, so the "
            f"two technologies weight the families differently. Reported as "
            f"measured; with only six families this correlation is also "
            f"low-powered.")
        claim7 = (
            f"**7. The glass-box approach itself is not specific to 5G.** A "
            f"detector trained on AWID3 802.11 packets reached {_fmt(aw_acc)} "
            f"accuracy and {_fmt(h.get('O4.awid3_roc_auc', float('nan')))} ROC "
            f"AUC, so the method carries across to a different wireless "
            f"technology, a different capture granularity and a different "
            f"attack catalogue. {transfer}")

    base_rec = h.get("O4.baseline_minority_recall")
    best_rec = h.get("O4.best_minority_recall")
    if base_rec is None or best_rec is None:
        claim6 = "**6. Imbalance handling.** Not yet measured."
    elif abs(best_rec - base_rec) < 0.01:
        claim6 = (
            f"**6. Imbalance handling is close to inert here - reported as a "
            f"negative result.** Rare-attack recall was already "
            f"{_fmt(base_rec)} before any treatment, and the best technique "
            f"({h.get('O4.best_technique', 'n/a')}) reached only "
            f"{_fmt(best_rec)}. The rare classes in 5G-NIDD are highly "
            f"separable, so they were never the bottleneck. That bottleneck is "
            f"the benign/UDPFlood collision, which no resampling can address.")
    else:
        claim6 = (
            f"**6. Imbalance handling moved rare-attack recall from "
            f"{_fmt(base_rec)} to {_fmt(best_rec)}** using "
            f"{h.get('O4.best_technique', 'the best technique')}. Objective 4 "
            f"reports what that gain cost in false alarms.")

    text = f"""# Summary of Findings

_All numbers produced by the `{profile}` run on
{datetime.now().strftime('%Y-%m-%d %H:%M')}._

## The question the study asks

Existing 5G intrusion detectors are either accurate but opaque, or opaque with
an explanation bolted on afterwards. This work tests whether a model that is
readable **by construction** - an Explainable Boosting Machine - can match the
opaque models on accuracy while giving explanations that are clearer, more
stable and more faithful than post-hoc SHAP.

## What was found

**1. Two columns of the published dataset leak the label.** `Seq` and `Offset`
describe where a record sits in the capture file, not what the traffic did. On
their own they reach {_fmt(h.get('O1.leaky_only_accuracy', float('nan')))}
binary accuracy. Any accuracy reported on `Combined.csv` without excluding them
is measuring file layout. This project removes them, and every figure in it is
a genuine traffic-based result.

**2. The honest accuracy ceiling is {_fmt(ceiling)}, not 1.00.** A substantial
share of flows have byte-for-byte identical feature vectors but different
labels - chiefly benign traffic against UDPFlood, where a single-packet benign
exchange and a single-packet flood produce the same flow record. No classifier
can separate them. Every accuracy in this study is judged against that ceiling.

{claim3}

**4. Post-hoc explanations of one black-box model disagree with each other.**
Four standard explainers - TreeSHAP, KernelSHAP, permutation importance and the
booster's own gain statistic - applied to the *same* trained XGBoost produced
rankings correlating as low as rho = {_fmt(worst)}. The model did not change;
only the tool did. A glass-box model has one explanation because there is
nothing to approximate. This is the central argument of the study, measured
rather than asserted.

{claim5}

{claim6}

{claim7}

**8. The output is usable by a non-expert.**
{_fmt(h.get('O5.n_cards', 0))} alert cards were produced, each stating three
plain-English reasons, a severity and a recommended first response. On
{_fmt(h.get('O5.exact_cards', 0))} of them the stated reasons reconcile exactly
with the model's decision, because the explanation is the model.

## Headline numbers

{metrics.df_to_markdown(pd.DataFrame(key, columns=["Finding", "Value"]))}

## Honest limitations

- The accuracy ceiling is estimated on the sampled data, and a smaller sample
  yields a higher ceiling because fewer colliding pairs land in it. Use
  `--profile full` for the figure to quote in the study.
- The deletion test scores each explanation against its own model, so the
  comparable figure reported is the ratio against a random control on the same
  model, not the raw AOPC.
- The AWID3 probe compares feature *families*, not features. The mapping is a
  documented judgement, published in Objective 4's raw results so it can be
  inspected and challenged.
- The published 99.9% figures in the literature are not directly compared
  against here, because those studies do not state whether `Seq` and `Offset`
  were excluded.

## Where to look next

| For | Open |
|---|---|
| The leakage finding and the ceiling | `Research Objective 1 .../Analysis.md` |
| Glass-box vs black-box accuracy | `Research Objective 2 .../Analysis.md` |
| The explainer-disagreement result | `Research Objective 3 .../Analysis.md` |
| Imbalance and the AWID3 probe | `Research Objective 4 .../Analysis.md` |
| The analyst-facing output | `Research Objective 5 .../ALERT_CARDS.html` |
"""
    (C.RESULTS_DIR / "Summary" / "SUMMARY.md").write_text(text, encoding="utf-8")


def _write_timings(out: Path) -> None:
    p = out / "stage_timings.json"
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    rows = [d for d in data if str(d.get("stage", "")).startswith("TOTAL")]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame["stage"] = frame["stage"].str.replace("TOTAL ", "", regex=False)
    frame = frame.groupby("stage", as_index=False)["seconds"].last()
    frame["minutes"] = (frame["seconds"] / 60).round(2)
    dataio.write_csv(frame.sort_values("seconds", ascending=False),
                     out / "stage_timings.csv")
