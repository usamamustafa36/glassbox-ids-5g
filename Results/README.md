# Results

**Explainable Intrusion Detection for 5G Wireless Networks Using a Glass-Box Machine Learning Model**  
Usama Mustafa

_Generated 2026-08-02 15:12 from the `full` profile (1,215,890 (all) flows). Random seed 42._

---

## How this folder is organised

One folder per research objective. Inside each:

| File | What it contains |
|---|---|
| `Results.md` | What was produced and what the numbers are |
| `Analysis.md` | What each metric means, what the values indicate, and how they demonstrate the objective |
| `raw_results/` | The underlying CSV and JSON outputs |
| `figures/` | Every chart, as PNG |

## The five research objectives

### Objective 1

> Prepare and profile the 5G-NIDD dataset and build baseline detectors, reporting both overall accuracy and the detection rate of the rare attack classes to expose the minority-class gap.

- [Results](Research%20Objective%201%20-%20Data%20Preparation%20and%20Baseline%20Detectors/Results.md)
- [Analysis](Research%20Objective%201%20-%20Data%20Preparation%20and%20Baseline%20Detectors/Analysis.md)

### Objective 2

> Develop a glass-box detector (EBM) in stages - binary attack detection first, then the harder multiclass case with its severe imbalance - and measure its accuracy against black-box baselines.

- [Results](Research%20Objective%202%20-%20Glass-Box%20(EBM)%20Detector/Results.md)
- [Analysis](Research%20Objective%202%20-%20Glass-Box%20(EBM)%20Detector/Analysis.md)

### Objective 3

> Compare the EBM's intrinsic explanations with post-hoc SHAP explanations of an XGBoost model on identical splits, in terms of clarity, stability and faithfulness.

- [Results](Research%20Objective%203%20-%20Explanation%20Quality%20Comparison/Results.md)
- [Analysis](Research%20Objective%203%20-%20Explanation%20Quality%20Comparison/Analysis.md)

### Objective 4

> Evaluate how imbalance-handling techniques affect minority-class recall, and run a generalisation probe on AWID3, examining whether the kinds of features the model relies on remain informative.

- [Results](Research%20Objective%204%20-%20Class%20Imbalance%20and%20Generalisation/Results.md)
- [Analysis](Research%20Objective%204%20-%20Class%20Imbalance%20and%20Generalisation/Analysis.md)

### Objective 5

> Produce a simple, analyst-readable explanation output suitable for a non-expert decision-maker.

- [Results](Research%20Objective%205%20-%20Analyst-Readable%20Explanations/Results.md)
- [Analysis](Research%20Objective%205%20-%20Analyst-Readable%20Explanations/Analysis.md)
- [**Alert cards (open in a browser)**](Research%20Objective%205%20-%20Analyst-Readable%20Explanations/ALERT_CARDS.html)

## Headline numbers

| Finding | Value |
|---|---|
| Class imbalance in 5G-NIDD (largest : smallest) | 413.6251 |
| Accuracy from capture position alone (the leakage found) | 0.9678 |
| Attainable accuracy ceiling once leakage is removed | 0.7685 |
| Glass-box (EBM) binary accuracy | 0.7684 |
| Best black-box binary accuracy | 0.7640 |
| p-value of the glass-box vs black-box difference | 0.0001 |
| Glass-box 9-class macro-F1 | 0.9241 |
| Worst agreement between two post-hoc explainers of ONE model | 0.5457 |
| EBM error against a known ground-truth effect | 0.1734 |
| SHAP error against the same known effect | 0.4113 |
| Best rare-attack recall after imbalance handling | 0.9983 |
| Glass-box accuracy on AWID3 (Wi-Fi) | 0.9860 |
| Feature-family transfer, 5G-NIDD to AWID3 | 0.2571 |
| Analyst alert cards produced | 8 |
| Cards whose reasons reconstruct the decision exactly | 8 |

See [`Summary/SUMMARY.md`](Summary/SUMMARY.md) for the cross-objective narrative and [`Summary/objective_validation.json`](Summary/objective_validation.json) for the objective coverage check.

## Reproducing these results

```bash
python -m pip install -r requirements.txt
python run.py verify
python run.py all --profile full
```

Full instructions: [`DEMO_GUIDE.md`](../DEMO_GUIDE.md)
