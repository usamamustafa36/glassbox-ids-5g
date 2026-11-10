"""Command-line interface driving the whole demonstration."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
import warnings
from typing import Callable

# Library chatter that would otherwise interrupt the demonstration. Genuine
# errors still surface as exceptions; only these known-benign notices are muted.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Missing values detected.*")
warnings.filterwarnings("ignore", message=".*n_quantiles.*")

from . import config as C
from . import console as ui

# name -> (module, objective(s), one-line description, runtime hint)
STAGES: dict[str, tuple[str, str, str, str]] = {
    "verify": ("step_verify", "-",
               "Check Python, packages, datasets and write access.", "~5 s"),
    "prepare": ("step_prepare", "1",
                "Clean and profile 5G-NIDD, audit for leakage, build the fixed split.",
                "~1-3 min"),
    "baselines": ("step_baselines", "1",
                  "Train the literature baseline detectors; expose the minority-class gap.",
                  "~2-5 min"),
    "glassbox": ("step_glassbox", "2",
                 "Train the EBM glass-box detector (binary then 9-class) against black boxes.",
                 "~9-15 min"),
    "explain": ("step_explain", "3",
                "Compare EBM intrinsic explanations with SHAP-on-XGBoost: clarity, stability, faithfulness.",
                "~3-7 min"),
    "imbalance": ("step_imbalance", "4",
                  "Measure how class weighting and SMOTE change minority-class recall.",
                  "~3-8 min"),
    "generalize": ("step_generalize", "4",
                   "Generalisation probe on AWID3 (Wi-Fi 802.11).", "~2-5 min"),
    "alerts": ("step_alerts", "5",
               "Produce analyst-readable alert cards a non-expert can act on.",
               "~1-2 min"),
    "report": ("step_report", "all",
               "Build the cross-objective summary and the Results index.", "~10 s"),
    "validate": ("step_validate", "all",
                 "Cross-check every research objective against the artifacts on disk.",
                 "~5 s"),
    "explain-flow": ("step_alerts", "5",
                     "Explain any single flow live, on demand (use --row N).",
                     "~10 s"),
}

#: The order used by `run.py all`.
PIPELINE = ["prepare", "baselines", "glassbox", "explain", "imbalance",
            "generalize", "alerts", "report", "validate"]


def _load(stage: str) -> Callable[..., int]:
    mod = importlib.import_module(f".steps.{STAGES[stage][0]}", package="xids5g")
    if stage == "explain-flow":
        return lambda profile, **kw: mod.explain_one(profile, kw.get("row", 0))
    return mod.run


def _intro(profile: str) -> None:
    prof = C.PROFILES[profile]
    ui.banner(
        "EXPLAINABLE INTRUSION DETECTION FOR 5G WIRELESS NETWORKS",
        "Glass-Box Machine Learning (EBM) vs Black-Box XGBoost + post-hoc SHAP\n"
        "Usama Mustafa",
    )
    ui.kv({
        "Run profile": f"{profile}  ({'all 1,215,890' if prof.n_rows is None else f'{prof.n_rows:,}'} flows)",
        "Expected runtime": prof.approx_runtime,
        "Primary dataset": "5G-NIDD (Samarakoon et al.)",
        "Probe dataset": "AWID3 (IEEE 802.11 Wi-Fi)",
        "Results folder": str(C.RESULTS_DIR.name) + "/",
        "Random seed": C.SEED,
    })
    ui.blank()


def _run_all(profile: str, args: argparse.Namespace) -> int:
    _intro(profile)
    ui.step(f"The pipeline will run {len(PIPELINE)} stages in order:")
    for i, s in enumerate(PIPELINE, 1):
        ui.info(f"{i}. {s:<12s} objective {STAGES[s][1]:<3s} {STAGES[s][2]}")
    ui.blank()

    t0 = time.perf_counter()
    done: list[tuple[str, float, int]] = []
    for i, stage in enumerate(PIPELINE, 1):
        ui.blank()
        ui.rule(f"[{i}/{len(PIPELINE)}]  {stage}")
        st = time.perf_counter()
        try:
            rc = _load(stage)(profile=profile, **_stage_kwargs(stage, args))
        except Exception:
            ui.fail(f"Stage '{stage}' failed:")
            traceback.print_exc()
            rc = 1
        dt = time.perf_counter() - st
        done.append((stage, dt, rc))
        if rc != 0 and stage not in ("validate",):
            ui.fail(f"Stopping: stage '{stage}' did not complete successfully.")
            break

    total = time.perf_counter() - t0
    ui.blank()
    ui.rule("Pipeline summary")
    ui.table("Stage timings", ["Stage", "Objective", "Status", "Duration"],
             [[s, STAGES[s][1], "OK" if rc == 0 else "FAILED",
               _fmt(d)] for s, d, rc in done])
    ui.kv({"Total wall-clock time": _fmt(total),
           "Results written to": str(C.RESULTS_DIR)})
    failed = [s for s, _, rc in done if rc != 0]
    if failed:
        ui.fail("Failed stages: " + ", ".join(failed))
        return 1
    ui.ok("Complete pipeline finished. Open the Results folder.")
    return 0


def _fmt(s: float) -> str:
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:02.0f}s"


def _stage_kwargs(stage: str, args: argparse.Namespace) -> dict:
    kw: dict = {}
    if stage == "alerts":
        kw["n_cards"] = getattr(args, "cards", 8)
    if stage == "generalize":
        kw["awid_rows"] = getattr(args, "awid_rows", 300_000)
    if stage == "explain-flow":
        kw["row"] = getattr(args, "row", 0)
    return kw


def build_parser() -> argparse.ArgumentParser:
    epilog = "Stages (run in this order, or use `all`):\n" + "\n".join(
        f"  {name:<12s} objective {meta[1]:<4s} {meta[3]:<9s} {meta[2]}"
        for name, meta in STAGES.items()
    ) + "\n\nFull walkthrough: see DEMO_GUIDE.md"

    p = argparse.ArgumentParser(
        prog="python run.py",
        description="Explainable Intrusion Detection for 5G - "
                    "glass-box (EBM) vs black-box (XGBoost + SHAP).",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("stage", choices=list(STAGES) + ["all"],
                   help="which pipeline stage to run ('all' runs everything)")
    p.add_argument("--profile", choices=list(C.PROFILES), default=C.DEFAULT_PROFILE,
                   help=f"how much data to use (default: {C.DEFAULT_PROFILE})")
    p.add_argument("--cards", type=int, default=8,
                   help="alerts stage: number of alert cards to generate")
    p.add_argument("--awid-rows", type=int, default=300_000, dest="awid_rows",
                   help="generalize stage: rows to sample from AWID3")
    p.add_argument("--row", type=int, default=0,
                   help="explain-flow stage: which test-split flow to explain")
    p.add_argument("--ascii", action="store_true",
                   help="plain ASCII output for terminals without Unicode support")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ascii:
        import os
        os.environ["XIDS_ASCII"] = "1"

    if args.stage == "all":
        rc = _run_all(args.profile, args)
    else:
        if args.stage != "verify":
            _intro(args.profile)
        try:
            rc = _load(args.stage)(profile=args.profile,
                                   **_stage_kwargs(args.stage, args))
        except FileNotFoundError as exc:
            ui.fail(str(exc))
            return 1
        except KeyboardInterrupt:                       # pragma: no cover
            ui.warn("Interrupted by user.")
            return 130

    ui.dump_timings(C.RESULTS_DIR / "Summary" / "stage_timings.json")
    return rc


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(main())
