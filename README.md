# Glass-box vs post-hoc explanations for 5G intrusion detection

**Usama Mustafa** · [github.com/usamamustafa36](https://github.com/usamamustafa36)

Machine-learning intrusion detectors for 5G are accurate but opaque. The usual fix is to explain a
black box after training with SHAP. This project tests the alternative: a **glass-box** detector, the
Explainable Boosting Machine (EBM), whose explanation *is* the model. It compares the two on identical
splits of **5G-NIDD** (1,215,890 flows), scored on accuracy, explanation stability, and explanation
faithfulness, plus a generalisation probe on **AWID3** (802.11).

## Findings

**1. Two columns of the public 5G-NIDD CSV leak the label.** `Seq` and `Offset` record where a flow
sits in the capture file, not what the traffic did. Because the CSV concatenates separate benign and
per-attack captures, those two columns alone reach **96.8% accuracy** with no traffic information at
all. Both are removed from every model here.

![Leakage audit](Results/Research%20Objective%201%20-%20Data%20Preparation%20and%20Baseline%20Detectors/figures/03_leakage_audit.png)

**2. The honest accuracy ceiling is 0.768, not ~0.999.** Many flows have identical feature vectors
but different labels (mainly benign vs UDPFlood), so no classifier can separate them. The pipeline
computes that ceiling exactly and reports every model against it. The EBM reaches **0.7684** against
a ceiling of 0.7685, and the best black box reaches 0.7640 (p = 6.5×10⁻⁵).

![Binary comparison](Results/Research%20Objective%202%20-%20Glass-Box%20(EBM)%20Detector/figures/20_binary_comparison.png)

**3. Post-hoc explanations disagree with each other and miss a known effect.** Two post-hoc
explainers of the *same* model agree as little as **0.55**. Against an injected ground-truth effect,
the EBM's error is **0.17** and SHAP's is **0.41**. The EBM wins 5 of 7 explanation-quality
comparisons. Seed stability is similar for both (0.94).

![Ground-truth recovery](Results/Research%20Objective%203%20-%20Explanation%20Quality%20Comparison/figures/34_ground_truth_recovery.png)
![Explainer disagreement](Results/Research%20Objective%203%20-%20Explanation%20Quality%20Comparison/figures/32_explainer_disagreement.png)

**4. Generalisation.** On AWID3 the glass-box detector reaches 0.986 accuracy (ROC-AUC 0.998), but
which *feature families* matter transfers only weakly from 5G-NIDD (Spearman 0.26).

Full results, one folder per question, each with `Results.md`, `Analysis.md`, raw CSV/JSON and
figures: [`Results/`](Results/README.md). Headline numbers:
[`Results/Summary/headline_numbers.json`](Results/Summary/headline_numbers.json).

## Run it

```bash
python -m pip install -r requirements.txt
python run.py verify                 # environment check
python run.py all --profile demo     # whole pipeline on a sample
python run.py all --profile full     # the figures reported above
```

Fixed seed (42). One stratified 60/20/20 split is computed once and reused by every model, so the
glass-box vs black-box comparison is like-for-like by construction.

## Data (not included)

- **5G-NIDD** (Samarakoon et al., 2022): place `Combined.csv` at `data/5g-nidd/Combined.csv`.
- **AWID3**: downloaded and prepared automatically by `python run.py generalize`.

## Layout

```
run.py              entry point
src/xids5g/         implementation (one module per pipeline stage in steps/)
Results/            all generated results and figures
```

MIT licence.
