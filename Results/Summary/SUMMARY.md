# Summary of Findings

_All numbers produced by the `full` run on
2026-08-02 15:12._

## The question the study asks

Existing 5G intrusion detectors are either accurate but opaque, or opaque with
an explanation bolted on afterwards. This work tests whether a model that is
readable **by construction** - an Explainable Boosting Machine - can match the
opaque models on accuracy while giving explanations that are clearer, more
stable and more faithful than post-hoc SHAP.

## What was found

**1. Two columns of the published dataset leak the label.** `Seq` and `Offset`
describe where a record sits in the capture file, not what the traffic did. On
their own they reach 0.9678
binary accuracy. Any accuracy reported on `Combined.csv` without excluding them
is measuring file layout. This project removes them, and every figure in it is
a genuine traffic-based result.

**2. The honest accuracy ceiling is 0.7685, not 1.00.** A substantial
share of flows have byte-for-byte identical feature vectors but different
labels - chiefly benign traffic against UDPFlood, where a single-packet benign
exchange and a single-packet flood produce the same flow record. No classifier
can separate them. Every accuracy in this study is judged against that ceiling.

**3. The glass-box model costs nothing in accuracy.** EBM scored 0.7684 against the best black box at 0.7640 - a difference of +0.0044, with a paired significance test over 5 independent re-trainings returning p = 0.0001. Both sit against the ceiling, so there is no accuracy left for a black box to claim. The accuracy-versus-interpretability trade-off does not appear on this data.

**4. Post-hoc explanations of one black-box model disagree with each other.**
Four standard explainers - TreeSHAP, KernelSHAP, permutation importance and the
booster's own gain statistic - applied to the *same* trained XGBoost produced
rankings correlating as low as rho = 0.5457. The model did not change;
only the tool did. A glass-box model has one explanation because there is
nothing to approximate. This is the central argument of the study, measured
rather than asserted.

**5. Against a known ground truth, the glass-box explanation is closer.** On synthetic data with a known generating function, the EBM shape function recovered the true effect with RMSE 0.1734 against 0.4113 for SHAP-on-XGBoost.

**6. Imbalance handling is close to inert here - reported as a negative result.** Rare-attack recall was already 0.9970 before any treatment, and the best technique (Class weighting) reached only 0.9983. The rare classes in 5G-NIDD are highly separable, so they were never the bottleneck. That bottleneck is the benign/UDPFlood collision, which no resampling can address.

**7. The glass-box approach itself is not specific to 5G.** A detector trained on AWID3 802.11 packets reached 0.9860 accuracy and 0.9985 ROC AUC, so the method carries across to a different wireless technology, a different capture granularity and a different attack catalogue. The feature-family ordering, however, transfers only weakly (rank correlation 0.2571, with 1 of the top 3 families shared). 802.11 headers carry radio-layer signal data with no 5G flow-record equivalent, and vice versa, so the two technologies weight the families differently. Reported as measured; with only six families this correlation is also low-powered.

**8. The output is usable by a non-expert.**
8 alert cards were produced, each stating three
plain-English reasons, a severity and a recommended first response. On
8 of them the stated reasons reconcile exactly
with the model's decision, because the explanation is the model.

## Headline numbers

| Finding | Value |
| --- | --- |
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
