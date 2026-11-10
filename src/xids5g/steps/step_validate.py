"""Stage 9 - objective coverage check.

Walks every research objective of this study, checks that the artifacts
which demonstrate it are actually on disk, and prints a pass/fail matrix. This
is the cross-check the project is graded against: it fails loudly if an
objective has been claimed but not evidenced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config as C
from .. import console as ui
from .. import dataio


@dataclass(frozen=True)
class Requirement:
    objective: int
    clause: str            # the objective wording this covers
    evidence: str          # what demonstrates it
    paths: tuple[str, ...]  # relative to the objective's Results folder


REQUIREMENTS: tuple[Requirement, ...] = (
    # ---------------------------------------------------------- Objective 1
    Requirement(1, "Prepare the 5G-NIDD dataset",
                "cleaning, encoding and the fixed stratified split",
                ("raw_results/04_removed_columns.csv",
                 "raw_results/05_feature_list.csv")),
    Requirement(1, "Profile the dataset",
                "class distribution and missing-value profile",
                ("raw_results/01_class_distribution.csv",
                 "raw_results/02_missing_values.csv",
                 "figures/01_class_distribution.png")),
    Requirement(1, "Build baseline detectors",
                "four literature baselines, binary and 9-class",
                ("raw_results/10_baselines_binary.csv",
                 "raw_results/11_baselines_multiclass.csv")),
    Requirement(1, "Report overall accuracy",
                "accuracy for every baseline on both tasks",
                ("raw_results/10_baselines_binary.csv",
                 "raw_results/11_baselines_multiclass.csv")),
    Requirement(1, "Report the detection rate of the rare attack classes",
                "per-class recall plus the aggregated minority gap",
                ("raw_results/12_minority_gap.csv",
                 "raw_results/13_per_class_recall.csv",
                 "figures/11_per_class_recall.png",
                 "figures/12_minority_gap.png")),
    # ---------------------------------------------------------- Objective 2
    Requirement(2, "Develop a glass-box (EBM) detector - binary stage",
                "trained EBM with binary results against black boxes",
                ("raw_results/20_binary_comparison.csv",
                 "figures/20_binary_comparison.png")),
    Requirement(2, "Develop a glass-box (EBM) detector - multiclass stage",
                "9-class EBM under the severe imbalance",
                ("raw_results/21_multiclass_comparison.csv",
                 "raw_results/22_multiclass_per_class.csv",
                 "figures/21_multiclass_comparison.png")),
    Requirement(2, "Benchmark accuracy against black-box baselines",
                "head-to-head comparison with a significance test",
                ("raw_results/23_significance.json",
                 "figures/24_run_variability.png")),
    Requirement(2, "Benchmark interpretability against black-box baselines",
                "audit of what can be read out of each model, plus shape functions",
                ("raw_results/24_interpretability_audit.json",
                 "raw_results/25_ebm_global_importance.csv",
                 "figures/25_ebm_global_importance.png")),
    # ---------------------------------------------------------- Objective 3
    Requirement(3, "Compare explanations on identical splits",
                "both explanations computed on the same held-out flows",
                ("raw_results/30_global_importance.csv",)),
    Requirement(3, "Compare on clarity",
                "concentration measures for both explanations",
                ("raw_results/31_clarity.json",
                 "figures/30_global_importance.png")),
    Requirement(3, "Compare on stability",
                "re-training, perturbation and cross-explainer agreement",
                ("raw_results/32_stability.csv",
                 "raw_results/33_explainer_agreement.json",
                 "figures/31_stability.png",
                 "figures/32_explainer_disagreement.png")),
    Requirement(3, "Compare on faithfulness",
                "exactness, deletion test and ground-truth recovery",
                ("raw_results/34_faithfulness.json",
                 "figures/33_deletion_curve.png",
                 "figures/34_ground_truth_recovery.png")),
    # ---------------------------------------------------------- Objective 4
    Requirement(4, "Evaluate how imbalance techniques affect minority recall",
                "five training regimes measured on the untouched test split",
                ("raw_results/40_imbalance_sweep.csv",
                 "raw_results/41_change_vs_baseline.csv",
                 "figures/40_imbalance_sweep.png",
                 "figures/42_recall_change.png")),
    Requirement(4, "Run a generalisation probe on AWID3",
                "a glass-box detector trained and scored on 802.11 data",
                ("raw_results/44_awid3_detection.csv",
                 "raw_results/45_awid3_importance.csv")),
    Requirement(4, "Examine whether the kinds of features remain informative",
                "feature-family transfer analysis with a rank correlation",
                ("raw_results/46_family_transfer.csv",
                 "raw_results/47_family_mapping.json",
                 "figures/44_family_transfer.png")),
    # ---------------------------------------------------------- Objective 5
    Requirement(5, "Produce a simple, analyst-readable explanation output",
                "alert cards in Markdown, HTML and chart form",
                ("ALERT_CARDS.md", "ALERT_CARDS.html",
                 "raw_results/50_alert_cards.csv")),
    Requirement(5, "Suitable for a non-expert decision-maker",
                "readability measurements of the generated text",
                ("raw_results/52_readability.json",)),
)


def run(profile: str = C.DEFAULT_PROFILE, **_) -> int:
    t = ui.StageTimer("Objective validation")
    ui.banner(
        "STAGE 9  -  RESEARCH OBJECTIVE VALIDATION",
        "Every objective of this study, checked against the artifacts "
        "actually present on disk.",
    )

    rows, by_obj = [], {}
    for r in REQUIREMENTS:
        base = C.ro_dir(r.objective)
        missing = [p for p in r.paths if not (base / p).exists()]
        ok = not missing
        by_obj.setdefault(r.objective, []).append(ok)
        rows.append([r.objective, r.clause, r.evidence,
                     "PASS" if ok else "MISSING",
                     "" if ok else missing[0]])

    for n in sorted(C.RO_DIRS):
        ui.blank()
        ui.rule(f"Research Objective {n}")
        ui.info(C.RO_TITLES[n])
        sub = [r for r in rows if r[0] == n]
        ui.table("", ["Objective clause", "Evidence produced", "Status",
                      "First missing file"],
                 [[r[1], r[2], r[3], r[4]] for r in sub])
        docs = [(C.ro_dir(n) / "Results.md").exists(),
                (C.ro_dir(n) / "Analysis.md").exists()]
        ui.kv({
            "Clauses satisfied": f"{sum(by_obj[n])} of {len(by_obj[n])}",
            "Results.md": "present" if docs[0] else "MISSING",
            "Analysis.md": "present" if docs[1] else "MISSING",
            "Raw result files": _count(C.ro_raw(n)),
            "Figures": _count(C.ro_fig(n)),
        })

    ui.blank()
    ui.rule("Summary")
    summary_rows = []
    for n in sorted(C.RO_DIRS):
        passed, total = sum(by_obj[n]), len(by_obj[n])
        docs_ok = ((C.ro_dir(n) / "Results.md").exists()
                   and (C.ro_dir(n) / "Analysis.md").exists())
        complete = passed == total and docs_ok
        summary_rows.append([
            f"Objective {n}", f"{passed}/{total}",
            _count(C.ro_raw(n)), _count(C.ro_fig(n)),
            "yes" if docs_ok else "no",
            "COMPLETE" if complete else "INCOMPLETE"])
    ui.table("Research objective coverage",
             ["Objective", "Clauses met", "Raw files", "Figures",
              "Docs written", "Status"], summary_rows)

    all_ok = all(r[-1] == "COMPLETE" for r in summary_rows)
    dataio.write_json({
        "profile": profile,
        "all_objectives_complete": all_ok,
        "objectives": {
            str(n): {
                "title": C.RO_TITLES[n],
                "clauses_met": sum(by_obj[n]),
                "clauses_total": len(by_obj[n]),
                "raw_files": _count(C.ro_raw(n)),
                "figures": _count(C.ro_fig(n)),
            } for n in sorted(C.RO_DIRS)},
        "requirements": [
            {"objective": r[0], "clause": r[1], "evidence": r[2],
             "status": r[3], "missing": r[4]} for r in rows],
    }, C.RESULTS_DIR / "Summary" / "objective_validation.json")
    ui.saved(C.RESULTS_DIR / "Summary" / "objective_validation.json")

    ui.blank()
    if all_ok:
        ui.ok("All five research objectives are implemented, evidenced and "
              "documented.")
        t.finish("5 of 5 objectives complete")
        return 0
    incomplete = [r[0] for r in summary_rows if r[-1] != "COMPLETE"]
    ui.fail("Incomplete: " + ", ".join(incomplete))
    ui.info("Run the missing stages - see `python run.py --help` for the order.")
    t.finish("validation failed")
    return 1


def _count(d: Path) -> int:
    return len([p for p in d.glob("*") if p.is_file()]) if d.exists() else 0
