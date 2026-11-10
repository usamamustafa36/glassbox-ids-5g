"""Stage 0 - environment and dataset verification.

Run this first on a new machine. It proves the installation is complete
before any long-running work starts, so a missing package is discovered in
five seconds rather than fifteen minutes in.
"""

from __future__ import annotations

import importlib
import platform
import sys

from .. import config as C
from .. import console as ui

REQUIRED = [
    ("numpy", "1.24"),
    ("pandas", "2.0"),
    ("scipy", "1.10"),
    ("sklearn", "1.3"),
    ("xgboost", "2.0"),
    ("interpret", "0.6"),
    ("imblearn", "0.12"),
    ("shap", "0.44"),
    ("matplotlib", "3.7"),
    ("pyarrow", "14.0"),
    ("joblib", "1.3"),
]

PACKAGE_HINT = {
    "sklearn": "scikit-learn",
    "interpret": "interpret-core",
    "imblearn": "imbalanced-learn",
}


def _version_tuple(v: str) -> tuple:
    parts = []
    for chunk in v.split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def run(profile: str = C.DEFAULT_PROFILE, **_) -> int:
    t = ui.StageTimer("Environment verification")
    ui.banner(
        "STAGE 0  -  ENVIRONMENT VERIFICATION",
        "Confirms Python, the required packages and the datasets are all in place.",
    )

    ui.step("Checking the Python interpreter")
    ui.kv({
        "Python": platform.python_version(),
        "Interpreter": sys.executable,
        "Platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "CPU cores in use": C.N_JOBS,
        "Project root": str(C.PROJECT_ROOT),
    })
    py_ok = sys.version_info >= (3, 9)
    (ui.ok if py_ok else ui.fail)(
        f"Python {platform.python_version()} "
        f"{'meets' if py_ok else 'is below'} the minimum of 3.9")
    ui.blank()

    ui.step(f"Checking {len(REQUIRED)} required packages")
    rows, missing, outdated = [], [], []
    for mod, min_v in REQUIRED:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "unknown")
            if v != "unknown" and _version_tuple(v) < _version_tuple(min_v):
                status, why = "OLD", f"needs >= {min_v}"
                outdated.append(mod)
            else:
                status, why = "OK", ""
            rows.append([PACKAGE_HINT.get(mod, mod), v, status, why])
        except Exception as exc:
            rows.append([PACKAGE_HINT.get(mod, mod), "-", "MISSING",
                         type(exc).__name__])
            missing.append(mod)
    ui.table("Package check", ["Package", "Installed", "Status", "Note"], rows)
    ui.blank()

    ui.step("Checking datasets")
    data_rows = []
    nidd_ok = C.NIDD_CSV.exists()
    data_rows.append([
        "5G-NIDD (primary)",
        "found" if nidd_ok else "MISSING",
        f"{C.NIDD_CSV.stat().st_size / 1e6:,.0f} MB" if nidd_ok else "-",
        str(C.NIDD_CSV.relative_to(C.PROJECT_ROOT)) if nidd_ok else str(C.NIDD_CSV),
    ])
    awid_ok = C.AWID3_PARQUET.exists()
    data_rows.append([
        "AWID3 (generalisation probe)",
        "found" if awid_ok else "not prepared",
        f"{C.AWID3_PARQUET.stat().st_size / 1e6:,.1f} MB" if awid_ok else "-",
        str(C.AWID3_PARQUET.relative_to(C.PROJECT_ROOT)) if awid_ok
        else "run: python run.py generalize  (prepares it automatically)",
    ])
    ui.table("Dataset check", ["Dataset", "Status", "Size", "Location"], data_rows)
    ui.blank()

    ui.step("Checking write access to the output folders")
    C.ensure_dirs()
    probe = C.ARTIFACTS_DIR / ".write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        ui.ok(f"Results folder writable: {C.RESULTS_DIR.relative_to(C.PROJECT_ROOT)}")
        write_ok = True
    except Exception as exc:
        ui.fail(f"Cannot write to {C.ARTIFACTS_DIR}: {exc}")
        write_ok = False
    ui.blank()

    ui.step("Run profiles available")
    ui.table("Profiles", ["Profile", "Flows used", "Approx. runtime", "Purpose"],
             [[p.name + (" (default)" if p.name == C.DEFAULT_PROFILE else ""),
               f"{p.n_rows:,}" if p.n_rows else "1,215,890 (all)",
               p.approx_runtime, p.description] for p in C.PROFILES.values()])
    ui.blank()

    problems = []
    if missing:
        problems.append(
            "Missing packages: " + ", ".join(PACKAGE_HINT.get(m, m) for m in missing))
    if outdated:
        problems.append(
            "Out-of-date packages: " + ", ".join(PACKAGE_HINT.get(m, m) for m in outdated))
    if not nidd_ok:
        problems.append(f"5G-NIDD not found at {C.NIDD_CSV}")
    if not py_ok:
        problems.append("Python 3.9 or newer is required")
    if not write_ok:
        problems.append("Output folders are not writable")

    if problems:
        for p in problems:
            ui.fail(p)
        ui.blank()
        ui.warn("Fix the above, then run:  python -m pip install -r requirements.txt")
        t.finish("environment NOT ready")
        return 1

    ui.ok("Environment is ready. Every requirement is satisfied.")
    ui.info("Next step:  python run.py prepare")
    t.finish("environment ready")
    return 0
