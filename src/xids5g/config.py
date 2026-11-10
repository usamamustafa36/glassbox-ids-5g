"""Central configuration: paths, run profiles, feature policy, RNG seed.

Every module imports its paths from here so the project can be moved or
copied to a Windows laptop without editing a single hard-coded path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths.  PROJECT_ROOT is resolved relative to this file, so the project works
# from any working directory and from any drive letter on Windows.
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
NIDD_CSV: Path = DATA_DIR / "5g-nidd" / "Combined.csv"
AWID3_DIR: Path = DATA_DIR / "awid3"
AWID3_PARQUET: Path = AWID3_DIR / "awid3_subset.parquet"

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"      # intermediate: splits, models
RESULTS_DIR: Path = PROJECT_ROOT / "Results"          # reader-facing deliverables

# Results/<objective folder> --------------------------------------------------
RO_DIRS: dict[int, str] = {
    1: "Research Objective 1 - Data Preparation and Baseline Detectors",
    2: "Research Objective 2 - Glass-Box (EBM) Detector",
    3: "Research Objective 3 - Explanation Quality Comparison",
    4: "Research Objective 4 - Class Imbalance and Generalisation",
    5: "Research Objective 5 - Analyst-Readable Explanations",
}

RO_TITLES: dict[int, str] = {
    1: "Prepare and profile the 5G-NIDD dataset and build baseline detectors, "
       "reporting both overall accuracy and the detection rate of the rare "
       "attack classes to expose the minority-class gap.",
    2: "Develop a glass-box detector (EBM) in stages - binary attack detection "
       "first, then the harder multiclass case with its severe imbalance - and "
       "measure its accuracy against black-box baselines.",
    3: "Compare the EBM's intrinsic explanations with post-hoc SHAP explanations "
       "of an XGBoost model on identical splits, in terms of clarity, stability "
       "and faithfulness.",
    4: "Evaluate how imbalance-handling techniques affect minority-class recall, "
       "and run a generalisation probe on AWID3, examining whether the kinds of "
       "features the model relies on remain informative.",
    5: "Produce a simple, analyst-readable explanation output suitable for a "
       "non-expert decision-maker.",
}


def ro_dir(n: int) -> Path:
    """Absolute path of the Results folder for research objective ``n``."""
    return RESULTS_DIR / RO_DIRS[n]


def ro_raw(n: int) -> Path:
    return ro_dir(n) / "raw_results"


def ro_fig(n: int) -> Path:
    return ro_dir(n) / "figures"


def ensure_dirs() -> None:
    """Create every output directory used by the pipeline."""
    for p in (ARTIFACTS_DIR, RESULTS_DIR, AWID3_DIR):
        p.mkdir(parents=True, exist_ok=True)
    for n in RO_DIRS:
        ro_raw(n).mkdir(parents=True, exist_ok=True)
        ro_fig(n).mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "Summary").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED: int = 42

#: Worker threads for XGBoost and scikit-learn. Capped deliberately: handing
#: every core to XGBoost's multiclass booster causes thread oversubscription
#: that made the 9-class fit roughly thirty times slower on a 20-core machine
#: (96 s against 3 s). Eight is also what a typical laptop has, so
#: timings stay predictable.
N_JOBS: int = max(1, min(8, os.cpu_count() or 2))

#: The EBM scales the other way - it is the slowest fit in the pipeline and
#: measurably faster with every core available (209 s against 279 s on a
#: 20-core machine). It therefore gets its own, uncapped budget.
N_JOBS_EBM: int = max(1, os.cpu_count() or 2)

# --------------------------------------------------------------------------
# Run profiles.  The full dataset is 1,215,890 flows; a live demonstration
# needs a size that finishes while the audience is still watching.  Every
# profile uses the *same* stratified sampling and the *same* split, so results
# are directly comparable across profiles - only the precision changes.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    name: str
    n_rows: int | None          # None -> use every row
    description: str
    approx_runtime: str


#: Runtimes are measured, not estimated: the `demo` profile completed in
#: 12 min 27 s on a 20-core machine. A 4-core laptop should expect roughly twice
#: that. `Results/Summary/stage_timings.csv` records the actual figures for
#: whatever machine the pipeline last ran on.
PROFILES: dict[str, Profile] = {
    "quick": Profile(
        "quick", 120_000,
        "Fast smoke-test of the whole pipeline.",
        "~8-15 minutes end to end",
    ),
    "demo": Profile(
        "demo", 200_000,
        "Recommended for a live demonstration: full statistical "
        "behaviour, comfortable runtime.",
        "~12-25 minutes end to end",
    ),
    "full": Profile(
        "full", None,
        "Every one of the 1,215,890 flows. Use for the final study numbers.",
        "~1.5-2.5 hours end to end",
    ),
}

DEFAULT_PROFILE = "demo"


# --------------------------------------------------------------------------
# Feature policy for 5G-NIDD
# --------------------------------------------------------------------------

# Label columns - never used as inputs.
LABEL_BINARY = "Label"
LABEL_MULTI = "Attack Type"
LABEL_TOOL = "Attack Tool"

#: Columns removed because they identify the *capture file* rather than the
#: traffic.  ``Seq`` is the Argus record sequence number and ``Offset`` the byte
#: offset inside the capture; because 5G-NIDD was assembled by concatenating
#: separate benign and per-attack captures, both encode the label directly.
#: The leakage audit in Research Objective 1 quantifies exactly how much.
LEAKAGE_COLUMNS = ["Unnamed: 0", "Seq", "Offset"]

#: ``Attack Tool`` is a second label (it names the tool that produced the
#: attack), so it is dropped from the inputs as well.
LABEL_COLUMNS = [LABEL_BINARY, LABEL_MULTI, LABEL_TOOL]

#: Categorical (nominal) inputs.
CATEGORICAL_COLUMNS = ["Proto", "sDSb", "dDSb", "Cause", "State"]

#: A column is dropped if this fraction of its training values is missing.
MAX_MISSING_RATE = 0.99
#: A binary "was this attribute observed?" indicator is added when the missing
#: rate is at least this high - missingness in flow records is informative
#: (e.g. no destination TTL means the peer never answered).
MISSING_INDICATOR_MIN_RATE = 0.005

#: The rare attack classes that overall accuracy hides.  Used throughout the
#: reporting to make the minority-class picture explicit.
MINORITY_CLASSES = [
    "ICMPFlood", "SYNFlood", "UDPScan", "SYNScan", "TCPConnectScan",
]

BENIGN_LABEL = "Benign"

# Split proportions (stratified on the 9-class Attack Type label).
TEST_SIZE = 0.20
VAL_SIZE = 0.20      # of the whole dataset, taken from the remaining 80%

# --------------------------------------------------------------------------
# Model hyper-parameters.  Chosen once, on the validation split, and then
# frozen so that every model is compared under identical conditions.
# --------------------------------------------------------------------------
EBM_PARAMS = dict(
    max_bins=128,
    outer_bags=4,
    learning_rate=0.05,
    max_rounds=3000,
    random_state=SEED,
)
EBM_BINARY_INTERACTIONS = 8      # pairwise terms; supported in the binary case
EBM_MULTI_INTERACTIONS = 0       # EBM interactions are unreliable for 9 classes

#: The binary and 9-class EBMs deliberately share one configuration; only the
#: interaction count differs. An earlier build used a cheaper multiclass budget
#: (2 bags, lr 0.1, 1000 rounds) which looked harmless at 90k training rows
#: (macro-F1 0.925) but underfitted badly at 180k (macro-F1 0.885, rare-class
#: recall 0.912 against 0.999 for XGBoost) - the smallest classes are the ones
#: that need the most boosting rounds, and early stopping on overall log-loss
#: cut them off. Measured at 180k training rows:
#:
#:     2 bags, lr 0.10, 1000 rounds ->  234 s, macro-F1 0.885, rare recall 0.912
#:     4 bags, lr 0.10, 1000 rounds ->  226 s, macro-F1 0.888, rare recall 0.916
#:     2 bags, lr 0.05, 3000 rounds ->  671 s, macro-F1 0.925, rare recall 0.999
#:
#: Bagging is parallelised and therefore nearly free; the boosting budget is
#: what matters. A tuning shortcut that silently changes the headline result at
#: a different data size is not worth the minutes it saves.

#: The imbalance sweep in Objective 4 fits five 9-class EBMs, one per technique,
#: some on resampled data three times the size. It uses a reduced budget - kept
#: identical across all five, so the comparison between techniques stays
#: like-for-like - and its absolute scores are consequently NOT comparable with
#: Objective 2's. Four outer bags are retained because two produced a
#: degenerate fit on Borderline-SMOTE output.
EBM_SWEEP_PARAMS = dict(outer_bags=4, learning_rate=0.1, max_rounds=1000)

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.15,
    tree_method="hist",
    random_state=SEED,
)

#: The multiclass booster fits one tree per class per round, so the binary
#: settings would cost nine times as much for no measurable gain. Validated on
#: the validation split: macro-F1 is unchanged to three decimal places.
XGB_MULTI_PARAMS = dict(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.2,
    tree_method="hist",
    random_state=SEED,
)

RF_PARAMS = dict(n_estimators=150, max_depth=None, random_state=SEED)
DT_PARAMS = dict(max_depth=12, random_state=SEED)
KNN_PARAMS = dict(n_neighbors=5)
#: k-NN prediction is O(n_test x n_train); cap the reference set so the live
#: demonstration does not stall on a baseline model.
KNN_MAX_TRAIN = 60_000
LR_PARAMS = dict(max_iter=1000, tol=1e-3, random_state=SEED)

#: Number of independent re-trainings used for the stability analysis (RO3)
#: and for the significance tests (RO2).
N_REPEATS = 5
