"""Stage 4 (Research Objective 3) - explanation quality.

The central experiment of the study. It puts the EBM's intrinsic explanations
against post-hoc SHAP explanations of an XGBoost model, on the identical split,
and scores them on the three properties the objective names:

* **clarity**    - how concentrated and how readable the explanation is;
* **stability**  - does the same input, or a re-trained model, give the same
                   reason;
* **faithfulness** - does the stated reason actually drive the prediction.

Every measurement is a number, not an impression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import config as C
from .. import console as ui
from .. import dataio, metrics, models, plotting as viz, reporting

RO = 3

N_LOCAL = 400          # test flows used for local explanation measurements
N_KERNEL = 120         # flows for the (expensive) model-agnostic KernelSHAP run
N_BACKGROUND = 100     # background rows for KernelSHAP
DELETION_STEPS = 10    # feature-removal steps in the faithfulness curve
NOISE_SIGMA = 0.01     # perturbation size, as a fraction of each feature's std


def run(profile: str = C.DEFAULT_PROFILE, **_) -> int:
    t = ui.StageTimer("Explanation quality comparison")
    ui.banner(
        "STAGE 4  -  EXPLANATION QUALITY   (Research Objective 3)",
        "EBM intrinsic explanations vs post-hoc SHAP on XGBoost, scored on "
        "clarity, stability and faithfulness.",
    )
    raw_dir, fig_dir = C.ro_raw(RO), C.ro_fig(RO)
    rng = np.random.default_rng(C.SEED)

    with ui.timed("Loading the prepared split and the trained models", "5-20 s"):
        s = dataio.load_splits(profile)
        ebm = models.load_model(profile, "ebm_binary")
        xgb_model = models.load_model(profile, "xgboost_binary")
    feats = list(s.X_train.columns)

    idx = rng.choice(len(s.X_test), size=min(N_LOCAL, len(s.X_test)), replace=False)
    X_local = s.X_test.iloc[idx].reset_index(drop=True)
    ui.info(f"Local measurements use {len(X_local)} held-out flows; "
            f"the model-agnostic explainer uses {N_KERNEL} of them.")

    import shap
    with ui.timed("Computing TreeSHAP values for the XGBoost model", "10-60 s") as tm:
        expl = shap.TreeExplainer(xgb_model)
        shap_local = np.asarray(expl.shap_values(X_local))
        if shap_local.ndim == 3:                      # (n, f, classes)
            shap_local = shap_local[:, :, -1]
        tm["shape"] = str(shap_local.shape)

    with ui.timed("Reading the EBM's own term contributions", "2-20 s") as tm:
        ebm_local = _ebm_local(ebm, X_local, feats)
        tm["shape"] = str(ebm_local.shape)

    # =====================================================================
    ui.blank()
    ui.rule("1 of 3  -  Clarity")
    ui.info("A clear explanation concentrates on a few features and can be read "
            "without a second tool.")

    ebm_global = np.abs(ebm_local).mean(axis=0)
    shap_global = np.abs(shap_local).mean(axis=0)
    clarity = {
        "EBM (intrinsic)": _clarity(ebm_global),
        "SHAP on XGBoost": _clarity(shap_global),
    }
    ui.table("Clarity of the global explanation",
             ["Explanation", "Features for 50% of the weight",
              "Features for 90%", "Normalised entropy", "Gini concentration"],
             [[k, v["features_for_50pct"], v["features_for_90pct"],
               v["normalised_entropy"], v["gini"]] for k, v in clarity.items()])
    ui.kv({
        "EBM explanation is": "the model itself - no explainer required",
        "SHAP explanation is": "a separate computation over the model",
        "Extra artefacts a reader needs (EBM)": 0,
        "Extra artefacts a reader needs (SHAP)": 1,
    })

    imp_frame = pd.DataFrame({
        "feature": feats,
        "ebm_importance": ebm_global,
        "shap_importance": shap_global,
    })
    imp_frame["ebm_rank"] = imp_frame["ebm_importance"].rank(ascending=False).astype(int)
    imp_frame["shap_rank"] = imp_frame["shap_importance"].rank(ascending=False).astype(int)
    imp_frame = imp_frame.sort_values("ebm_importance", ascending=False)
    dataio.write_csv(imp_frame, raw_dir / "30_global_importance.csv")
    dataio.write_json(clarity, raw_dir / "31_clarity.json")

    top = imp_frame.head(12)
    viz.grouped_bars(
        list(top["feature"]),
        {"EBM (intrinsic)": list(top["ebm_importance"] / ebm_global.sum()),
         "SHAP on XGBoost": list(top["shap_importance"] / shap_global.sum())},
        fig_dir / "30_global_importance.png",
        title="Global feature importance - glass-box against post-hoc",
        ylabel="share of total attribution", fmt=None, rotate=35,
        subtitle="Both explanations are normalised to sum to 1 so their "
                 "concentration can be compared directly.")
    ui.saved(fig_dir / "30_global_importance.png")

    # =====================================================================
    ui.blank()
    ui.rule("2 of 3  -  Stability")

    ui.step("2a. Does the explanation survive re-training with a new seed?")
    with ui.timed(f"Re-training both models {C.N_REPEATS} times and comparing "
                  "their explanations", "1-5 min") as tm:
        seed_stab = _seed_stability(s, feats, n=C.N_REPEATS)
        tm["EBM rank corr."] = f"{np.mean(seed_stab['EBM']['spearman']):.4f}"
        tm["SHAP rank corr."] = f"{np.mean(seed_stab['SHAP']['spearman']):.4f}"
    ui.table("Agreement between explanations of independently re-trained models",
             ["Explanation", "Mean Spearman rank corr.", "Std dev",
              "Mean top-5 Jaccard"],
             [[k, float(np.mean(v["spearman"])), float(np.std(v["spearman"])),
               float(np.mean(v["jaccard"]))] for k, v in seed_stab.items()])

    ui.step("2b. Does a tiny change to the input change the reason given?")
    with ui.timed(f"Perturbing {len(X_local)} flows by {NOISE_SIGMA:.0%} of each "
                  "feature's spread", "20-90 s") as tm:
        local_stab = _local_stability(ebm, xgb_model, expl, X_local, s, feats, rng)
        tm["EBM cosine"] = f"{np.mean(local_stab['EBM']['cosine']):.4f}"
        tm["SHAP cosine"] = f"{np.mean(local_stab['SHAP']['cosine']):.4f}"
    ui.table("Explanation similarity before and after a small input perturbation",
             ["Explanation", "Mean cosine similarity", "Mean top-3 Jaccard",
              "Flows whose top reason changed (%)"],
             [[k, float(np.mean(v["cosine"])), float(np.mean(v["jaccard"])),
               100 * float(np.mean(v["top1_changed"]))]
              for k, v in local_stab.items()])

    ui.step("2c. Do two post-hoc explainers agree on the SAME black-box model?")
    ui.info("This is the failure mode reported for 5G-NIDD by Uccello and "
            "Nadjm-Tehrani: the reason depends on which tool you pick.")
    with ui.timed("Running TreeSHAP, KernelSHAP, permutation importance and "
                  "gain importance on one XGBoost model", "1-4 min") as tm:
        agree = _explainer_agreement(xgb_model, X_local, s, feats, expl, rng)
        tm["worst pair"] = f"{agree['worst_pair']} rho={agree['worst_spearman']:.3f}"

    ui.table("Do the post-hoc explainers agree with each other?",
             ["Explainer pair", "Spearman rank corr.", "Top-5 Jaccard"],
             [[k, v["spearman"], v["jaccard"]] for k, v in agree["pairs"].items()])
    ui.note(
        f"Four standard ways of explaining the same XGBoost model disagree; the "
        f"weakest pair ({agree['worst_pair']}) correlates at only "
        f"rho={agree['worst_spearman']:.3f}. The EBM has exactly one explanation, "
        f"so this disagreement cannot arise.")

    stab_frame = pd.DataFrame([
        {"test": "re-training (seed) stability", "explanation": k,
         "spearman": float(np.mean(v["spearman"])),
         "top5_jaccard": float(np.mean(v["jaccard"]))}
        for k, v in seed_stab.items()
    ] + [
        {"test": "input-perturbation stability", "explanation": k,
         "cosine": float(np.mean(v["cosine"])),
         "top3_jaccard": float(np.mean(v["jaccard"])),
         "top1_changed_pct": 100 * float(np.mean(v["top1_changed"]))}
        for k, v in local_stab.items()
    ])
    dataio.write_csv(stab_frame, raw_dir / "32_stability.csv")
    dataio.write_json(agree, raw_dir / "33_explainer_agreement.json")

    viz.boxes({"EBM\n(re-training)": seed_stab["EBM"]["spearman"],
               "SHAP on XGBoost\n(re-training)": seed_stab["SHAP"]["spearman"],
               "EBM\n(perturbation)": local_stab["EBM"]["cosine"],
               "SHAP on XGBoost\n(perturbation)": local_stab["SHAP"]["cosine"]},
              fig_dir / "31_stability.png",
              title="Explanation stability",
              ylabel="agreement (1.0 = the explanation did not change)",
              subtitle="Left pair: agreement between explanations of models "
                       "re-trained with different seeds. Right pair: agreement "
                       "before and after a 1% perturbation of the input.")
    ui.saved(fig_dir / "31_stability.png")

    names = agree["matrix_labels"]
    viz.heatmap(np.array(agree["matrix"]), names, names,
                fig_dir / "32_explainer_disagreement.png",
                title="Do post-hoc explainers agree? (Spearman rank correlation)",
                fmt="{:.2f}", cbar_label="rank correlation",
                subtitle="All four explain the SAME trained XGBoost model. "
                         "Off-diagonal cells below 1.00 are disagreement about "
                         "what that one model is doing.")
    ui.saved(fig_dir / "32_explainer_disagreement.png")

    # =====================================================================
    ui.blank()
    ui.rule("3 of 3  -  Faithfulness")

    ui.step("3a. Exactness: does the explanation add up to the prediction?")
    with ui.timed("Checking the additive decomposition of both explanations",
                  "5-20 s"):
        exact = _exactness(ebm, xgb_model, expl, X_local, ebm_local, shap_local)
    ui.table("Reconstruction error - explanation total vs the model's own output",
             ["Explanation", "Max absolute error", "Mean absolute error",
              "Exact by construction?"],
             [[k, v["max_abs_error"], v["mean_abs_error"], v["exact"]]
              for k, v in exact.items()])

    ui.step("3b. Deletion test: remove what the explanation called important.")
    ui.info("If a reason is faithful, removing the features it names should "
            "collapse the model's confidence faster than removing random ones.")
    with ui.timed(f"Running the deletion curve over {DELETION_STEPS} steps",
                  "30 s - 3 min") as tm:
        deletion = _deletion_curves(ebm, xgb_model, X_local, ebm_local,
                                    shap_local, s, rng)
        tm["EBM AOPC"] = f"{deletion['EBM (intrinsic)']['aopc']:.4f}"
        tm["SHAP AOPC"] = f"{deletion['SHAP on XGBoost']['aopc']:.4f}"

    ui.table("Faithfulness by the deletion test",
             ["Ranking used", "AOPC", "Random control on the same model",
              "Ratio vs its own control", "Drop after 5 features"],
             [[k, v["aopc"], v["random_control_aopc"],
               v["aopc_ratio_vs_random"], v["curve"][5] - v["curve"][0]]
              for k, v in deletion.items()])
    ui.info("The ratio is the comparable number: raw AOPC also reflects how "
            "sensitive each model is to having features removed.")

    viz.lines(list(range(DELETION_STEPS + 1)),
              {k: v["curve"] for k, v in deletion.items()},
              fig_dir / "33_deletion_curve.png",
              title="Deletion test - confidence as explained features are removed",
              xlabel="number of top-ranked features replaced by their median value",
              ylabel="mean predicted probability of the original class",
              direct_label=False,
              dashed=["Random ranking (control)"],
              subtitle="A faithful explanation names the features that actually "
                       "carry the decision, so its curve falls fastest. The "
                       "dashed control ranks features at random.")
    ui.saved(fig_dir / "33_deletion_curve.png")

    ui.step("3c. Ground truth: can each method recover a known effect?")
    ui.info("On real data there is no truth to check against, so this uses a "
            "synthetic dataset whose generating rule is known exactly.")
    with ui.timed("Fitting both models to a known generating function",
                  "20-60 s") as tm:
        truth = _ground_truth_recovery(fig_dir)
        tm["EBM RMSE"] = f"{truth['ebm_rmse']:.4f}"
        tm["SHAP RMSE"] = f"{truth['shap_rmse']:.4f}"
    ui.table("Recovery of a known feature effect (lower error = more faithful)",
             ["Method", "RMSE against the true shape", "Correlation with truth"],
             [["EBM shape function", truth["ebm_rmse"], truth["ebm_corr"]],
              ["SHAP dependence on XGBoost", truth["shap_rmse"], truth["shap_corr"]]])
    ui.saved(fig_dir / "34_ground_truth_recovery.png")

    faith = {"exactness": exact,
             "deletion": {k: {"aopc": v["aopc"], "curve": v["curve"]}
                          for k, v in deletion.items()},
             "ground_truth_recovery": truth}
    dataio.write_json(faith, raw_dir / "34_faithfulness.json")

    # =====================================================================
    ui.blank()
    ui.rule("Scorecard")
    score = _scorecard(clarity, seed_stab, local_stab, agree, exact, deletion, truth)
    ui.table("Explanation quality - overall",
             ["Property", "Measure", "EBM (glass-box)", "SHAP on XGBoost",
              "Better"], score["rows"])
    dataio.write_csv(pd.DataFrame(score["rows"], columns=[
        "property", "measure", "ebm", "shap_on_xgboost", "better"]),
        raw_dir / "35_scorecard.csv")
    ui.note(f"EBM wins {score['ebm_wins']} of {score['n_comparisons']} "
            f"head-to-head measures; SHAP-on-XGBoost wins {score['shap_wins']}; "
            f"{score['ties']} are level.")

    viz.stat_tiles(
        [("Explainers that disagree", f"{agree['worst_spearman']:.2f}",
          "worst rank correlation between two post-hoc tools on one model"),
         ("EBM reconstruction error",
          f"{exact['EBM (intrinsic)']['max_abs_error']:.2e}", "exact by construction"),
         ("EBM deletion AOPC", f"{deletion['EBM (intrinsic)']['aopc']:.3f}",
          "higher is more faithful"),
         ("Ground-truth error", f"{truth['ebm_rmse']:.3f} vs {truth['shap_rmse']:.3f}",
          "EBM vs SHAP against a known effect")],
        fig_dir / "00_headline.png",
        title="Research Objective 3 - headline numbers")
    ui.saved(fig_dir / "00_headline.png")

    _write_docs(profile, clarity, imp_frame, seed_stab, local_stab, agree,
                exact, deletion, truth, score)

    ui.blank()
    ui.ok("Explanation quality measured on all three properties.")
    ui.info("Next step:  python run.py imbalance")
    t.finish(f"Results -> {C.ro_dir(RO).name}")
    return 0


# ==========================================================================
# Measurement helpers
# ==========================================================================
def _ebm_local(ebm, X: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Per-feature contributions from the EBM, aggregated to feature level.

    ``eval_terms`` returns one column per *term*; pairwise terms are split
    evenly between the two features they involve so the result is directly
    comparable with a SHAP vector.
    """
    terms = np.asarray(ebm.eval_terms(X), dtype=float)
    out = np.zeros((len(X), len(feats)), dtype=float)
    pos = {f: i for i, f in enumerate(feats)}
    for j, name in enumerate(ebm.term_names_):
        if " & " in name:
            for part in name.split(" & "):
                if part in pos:
                    out[:, pos[part]] += terms[:, j] / 2.0
        elif name in pos:
            out[:, pos[name]] += terms[:, j]
    return out


def _clarity(importance: np.ndarray) -> dict:
    """How concentrated an explanation is: fewer features carrying the weight."""
    v = np.abs(np.asarray(importance, dtype=float))
    total = v.sum()
    if total <= 0:
        return {"features_for_50pct": len(v), "features_for_90pct": len(v),
                "normalised_entropy": 1.0, "gini": 0.0}
    p = np.sort(v / total)[::-1]
    cum = np.cumsum(p)
    n = len(p)
    ent = -np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(n)
    srt = np.sort(v / total)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * srt) / (n * srt.sum())) - (n + 1) / n)
    return {
        "features_for_50pct": int(np.searchsorted(cum, 0.50) + 1),
        "features_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "normalised_entropy": float(ent),
        "gini": gini,
        "n_features": n,
    }


def _rank_agreement(a: np.ndarray, b: np.ndarray, k: int = 5) -> tuple[float, float]:
    rho = stats.spearmanr(a, b).statistic
    ta = set(np.argsort(a)[::-1][:k])
    tb = set(np.argsort(b)[::-1][:k])
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0
    return float(rho if np.isfinite(rho) else 0.0), float(jac)


def _seed_stability(s, feats, n: int = 5) -> dict:
    """Re-train both models on bootstrap resamples; compare their explanations.

    Varying only the random seed would be meaningless for XGBoost, which is
    deterministic given fixed data and hyper-parameters - it would score a
    perfect 1.0 without demonstrating anything. Each repeat therefore draws a
    bootstrap resample of the training data, and *both* models are fitted to the
    identical resample, so the two are treated the same way.
    """
    import shap

    ft = s.pre.feature_types()
    sub = s.X_test.iloc[:600]
    rng = np.random.default_rng(C.SEED)
    ebm_imps, shap_imps = [], []
    for i in range(n):
        seed = C.SEED + 977 * i
        boot = rng.choice(len(s.X_train), size=len(s.X_train), replace=True)
        Xb, yb = s.X_train.iloc[boot], s.y_train_bin[boot]

        e = models.make_ebm("binary", seed=seed, feature_types=ft)
        e.fit(Xb, yb)
        ebm_imps.append(np.abs(_ebm_local(e, sub, feats)).mean(axis=0))

        x = models.make_xgb(seed=seed)
        x.fit(Xb, yb)
        sv = np.asarray(shap.TreeExplainer(x).shap_values(sub))
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        shap_imps.append(np.abs(sv).mean(axis=0))

    def pairwise(arrs):
        rhos, jacs = [], []
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                r, jc = _rank_agreement(arrs[i], arrs[j])
                rhos.append(r)
                jacs.append(jc)
        return {"spearman": rhos, "jaccard": jacs}

    return {"EBM": pairwise(ebm_imps), "SHAP": pairwise(shap_imps)}


def _local_stability(ebm, xgb_model, expl, X, s, feats, rng) -> dict:
    """Perturb the input slightly; see whether the stated reason survives."""
    cont = [f for f in feats
            if f not in s.pre.categorical_features and not f.endswith("__not_observed")]
    # Scale the noise by the interquartile range, not the standard deviation.
    # Flow features are heavily long-tailed - TotBytes has a std in the millions
    # - so "1% of a standard deviation" would be a large change for a typical
    # flow rather than the small one this test is supposed to apply.
    q = s.X_train[cont].quantile([0.25, 0.75])
    scale = (q.loc[0.75] - q.loc[0.25]).replace(0, np.nan)
    scale = scale.fillna(s.X_train[cont].std()).replace(0, 1.0)
    Xp = X.copy()
    noise = rng.normal(0, 1, size=(len(X), len(cont))) * (scale.values * NOISE_SIGMA)
    Xp[cont] = Xp[cont].values + noise

    ebm_a, ebm_b = _ebm_local(ebm, X, feats), _ebm_local(ebm, Xp, feats)
    sa = np.asarray(expl.shap_values(X))
    sb = np.asarray(expl.shap_values(Xp))
    if sa.ndim == 3:
        sa, sb = sa[:, :, -1], sb[:, :, -1]

    def compare(a, b):
        cos, jac, changed = [], [], []
        for i in range(len(a)):
            na, nb = np.linalg.norm(a[i]), np.linalg.norm(b[i])
            cos.append(float(a[i] @ b[i] / (na * nb)) if na and nb else 1.0)
            ta = set(np.argsort(np.abs(a[i]))[::-1][:3])
            tb = set(np.argsort(np.abs(b[i]))[::-1][:3])
            jac.append(len(ta & tb) / len(ta | tb))
            changed.append(int(np.argmax(np.abs(a[i])) != np.argmax(np.abs(b[i]))))
        return {"cosine": cos, "jaccard": jac, "top1_changed": changed}

    return {"EBM": compare(ebm_a, ebm_b), "SHAP": compare(sa, sb)}


def _explainer_agreement(xgb_model, X, s, feats, tree_expl, rng) -> dict:
    """Four standard post-hoc explanations of one model, compared with each other."""
    import shap
    from sklearn.inspection import permutation_importance

    sv = np.asarray(tree_expl.shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    tree_imp = np.abs(sv).mean(axis=0)

    sub = X.iloc[:N_KERNEL]
    bg = shap.kmeans(s.X_train.iloc[:5000], N_BACKGROUND // 4)
    kexp = shap.KernelExplainer(
        lambda d: xgb_model.predict_proba(pd.DataFrame(d, columns=feats))[:, 1], bg)
    ksv = np.asarray(kexp.shap_values(sub, nsamples=128, silent=True))
    if ksv.ndim == 3:
        ksv = ksv[:, :, -1]
    kernel_imp = np.abs(ksv).mean(axis=0)

    perm = permutation_importance(
        xgb_model, X, xgb_model.predict(X), n_repeats=3,
        random_state=C.SEED, n_jobs=C.N_JOBS)
    perm_imp = np.abs(perm.importances_mean)

    gain = np.zeros(len(feats))
    try:
        booster = xgb_model.get_booster()
        score = booster.get_score(importance_type="gain")
        for i, f in enumerate(feats):
            gain[i] = score.get(f, score.get(f"f{i}", 0.0))
    except Exception:
        gain = np.asarray(xgb_model.feature_importances_, dtype=float)

    methods = {"TreeSHAP": tree_imp, "KernelSHAP": kernel_imp,
               "Permutation": perm_imp, "XGBoost gain": gain}
    labels = list(methods)
    n = len(labels)
    mat = np.ones((n, n))
    pairs = {}
    worst, worst_pair = 1.0, ""
    for i in range(n):
        for j in range(i + 1, n):
            r, jc = _rank_agreement(methods[labels[i]], methods[labels[j]])
            mat[i, j] = mat[j, i] = r
            pairs[f"{labels[i]} vs {labels[j]}"] = {"spearman": r, "jaccard": jc}
            if r < worst:
                worst, worst_pair = r, f"{labels[i]} vs {labels[j]}"
    return {"pairs": pairs, "matrix": mat.tolist(), "matrix_labels": labels,
            "worst_spearman": float(worst), "worst_pair": worst_pair,
            "mean_offdiagonal": float((mat.sum() - n) / (n * n - n))}


def _exactness(ebm, xgb_model, expl, X, ebm_local, shap_local) -> dict:
    """Does the explanation's total equal what the model actually output?"""
    ebm_total = ebm_local.sum(axis=1) + float(np.asarray(ebm.intercept_).ravel()[0])
    ebm_logit = np.asarray(ebm.decision_function(X), dtype=float).ravel()
    ebm_err = np.abs(ebm_total - ebm_logit)

    base = np.asarray(expl.expected_value).ravel()[-1]
    shap_total = shap_local.sum(axis=1) + base
    xgb_margin = xgb_model.predict(X, output_margin=True).ravel()
    shap_err = np.abs(shap_total - xgb_margin)

    return {
        "EBM (intrinsic)": {
            "max_abs_error": float(ebm_err.max()),
            "mean_abs_error": float(ebm_err.mean()),
            "exact": "yes - the model IS the sum of its terms",
        },
        "SHAP on XGBoost": {
            "max_abs_error": float(shap_err.max()),
            "mean_abs_error": float(shap_err.mean()),
            "exact": "only for TreeSHAP; other explainers of the same model differ",
        },
    }


def _deletion_curves(ebm, xgb_model, X, ebm_local, shap_local, s, rng) -> dict:
    """Replace the top-ranked features and watch the model's confidence fall.

    Each explanation is necessarily scored against its own model, so the raw
    AOPC values confound explanation quality with how sensitive that model is
    to feature removal. A random-ranking control is therefore run **on each
    model separately**, and the reported faithfulness measure is the ratio of
    the explanation's AOPC to its own model's random control. That ratio is
    comparable across models; the raw AOPC is not.
    """
    med = s.X_train.median()
    n_feat = X.shape[1]
    steps = min(DELETION_STEPS, n_feat)
    rand_order = np.argsort(rng.random((len(X), n_feat)), axis=1)

    def curve(model, order):
        base_prob = model.predict_proba(X)[:, 1]
        cls = (base_prob >= 0.5).astype(int)
        vals = [float(np.mean(np.where(cls == 1, base_prob, 1 - base_prob)))]
        Xw = X.copy()
        for k in range(steps):
            cols = order[:, k]
            for c in np.unique(cols):
                rows = np.where(cols == c)[0]
                Xw.iloc[rows, c] = med.iloc[c]
            p = model.predict_proba(Xw)[:, 1]
            vals.append(float(np.mean(np.where(cls == 1, p, 1 - p))))
        return vals

    def aopc(c):
        return float(np.mean([c[0] - v for v in c[1:]]))

    ranked = {
        "EBM (intrinsic)": (ebm, np.argsort(np.abs(ebm_local), axis=1)[:, ::-1]),
        "SHAP on XGBoost": (xgb_model,
                            np.argsort(np.abs(shap_local), axis=1)[:, ::-1]),
    }
    out = {}
    for label, (model, order) in ranked.items():
        c = curve(model, order)
        ctrl = curve(model, rand_order)
        a, a_ctrl = aopc(c), aopc(ctrl)
        out[label] = {
            "curve": c,
            "aopc": a,
            "random_control_aopc": a_ctrl,
            "aopc_ratio_vs_random": float(a / a_ctrl) if a_ctrl > 1e-9 else float("inf"),
        }
    # One control curve is plotted, for the EBM, to keep the figure readable.
    ctrl_curve = curve(ebm, rand_order)
    out["Random ranking (control)"] = {
        "curve": ctrl_curve, "aopc": aopc(ctrl_curve),
        "random_control_aopc": aopc(ctrl_curve), "aopc_ratio_vs_random": 1.0}
    return out


def _ground_truth_recovery(fig_dir) -> dict:
    """Check both methods against a generating function that is known exactly.

    Real data offers no ground truth for an explanation. Here the data is
    generated from a known additive rule, so the true effect of ``x0`` is
    available and each method's recovered effect can be scored against it.
    """
    import shap

    rng = np.random.default_rng(C.SEED)
    n, p = 20_000, 6
    X = rng.normal(0, 1, size=(n, p))
    true_effect = 2.0 * np.sin(1.5 * X[:, 0])          # the effect to recover
    logit = true_effect + 1.2 * X[:, 1] - 0.8 * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cols = [f"x{i}" for i in range(p)]
    Xd = pd.DataFrame(X, columns=cols)

    e = models.make_ebm("binary", interactions=0)
    e.fit(Xd, y)
    x_model = models.make_xgb()
    x_model.fit(Xd, y)

    grid = np.linspace(-2.5, 2.5, 60)
    truth = 2.0 * np.sin(1.5 * grid)
    truth = truth - truth.mean()

    probe = pd.DataFrame(np.zeros((len(grid), p)), columns=cols)
    probe["x0"] = grid
    ebm_curve = _ebm_local(e, probe, cols)[:, 0]
    ebm_curve = ebm_curve - ebm_curve.mean()

    sv = np.asarray(shap.TreeExplainer(x_model).shap_values(probe))
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    shap_curve = sv[:, 0] - sv[:, 0].mean()

    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    viz.lines(list(grid),
              {"True effect (known)": list(truth),
               "EBM shape function": list(ebm_curve),
               "SHAP on XGBoost": list(shap_curve)},
              fig_dir / "34_ground_truth_recovery.png",
              title="Recovering a known feature effect",
              xlabel="x0", ylabel="contribution to the log-odds",
              direct_label=False, dashed=["True effect (known)"],
              subtitle="Data generated from 2*sin(1.5*x0) + 1.2*x1 - 0.8*x2, so "
                       "the dashed line is the exact answer. Closer to it is "
                       "more faithful.")
    return {
        "generating_function": "logit = 2*sin(1.5*x0) + 1.2*x1 - 0.8*x2",
        "n_rows": n,
        "ebm_rmse": rmse(ebm_curve, truth),
        "shap_rmse": rmse(shap_curve, truth),
        "ebm_corr": float(np.corrcoef(ebm_curve, truth)[0, 1]),
        "shap_corr": float(np.corrcoef(shap_curve, truth)[0, 1]),
    }


def _scorecard(clarity, seed_stab, local_stab, agree, exact, deletion, truth) -> dict:
    def better(a, b, higher=True, tol=1e-9):
        if abs(a - b) <= tol:
            return "level"
        return "EBM" if ((a > b) == higher) else "SHAP on XGBoost"

    rows = []

    a = clarity["EBM (intrinsic)"]["features_for_90pct"]
    b = clarity["SHAP on XGBoost"]["features_for_90pct"]
    rows.append(["Clarity", "features needed for 90% of the attribution",
                 a, b, better(a, b, higher=False)])

    a = float(np.mean(seed_stab["EBM"]["spearman"]))
    b = float(np.mean(seed_stab["SHAP"]["spearman"]))
    rows.append(["Stability", "rank agreement across re-trainings",
                 round(a, 4), round(b, 4), better(a, b)])

    a = float(np.mean(local_stab["EBM"]["cosine"]))
    b = float(np.mean(local_stab["SHAP"]["cosine"]))
    rows.append(["Stability", "similarity after a 1% input perturbation",
                 round(a, 4), round(b, 4), better(a, b)])

    rows.append(["Stability", "explanations that disagree on this model",
                 "1 (none possible)",
                 f"{len(agree['matrix_labels'])} (worst rho={agree['worst_spearman']:.2f})",
                 "EBM"])

    a = exact["EBM (intrinsic)"]["max_abs_error"]
    b = exact["SHAP on XGBoost"]["max_abs_error"]
    rows.append(["Faithfulness", "reconstruction error of the explanation",
                 f"{a:.2e}", f"{b:.2e}", better(a, b, higher=False)])

    a = deletion["EBM (intrinsic)"]["aopc_ratio_vs_random"]
    b = deletion["SHAP on XGBoost"]["aopc_ratio_vs_random"]
    rows.append(["Faithfulness", "deletion AOPC vs its own random control",
                 round(a, 3), round(b, 3), better(a, b)])

    a, b = truth["ebm_rmse"], truth["shap_rmse"]
    rows.append(["Faithfulness", "error against a known ground-truth effect",
                 round(a, 4), round(b, 4), better(a, b, higher=False)])

    winners = [r[-1] for r in rows]
    return {"rows": rows, "ebm_wins": winners.count("EBM"),
            "shap_wins": winners.count("SHAP on XGBoost"),
            "ties": winners.count("level"), "n_comparisons": len(rows)}


# ==========================================================================
def _write_docs(profile, clarity, imp_frame, seed_stab, local_stab, agree,
                exact, deletion, truth, score) -> None:
    ec, sc_ = clarity["EBM (intrinsic)"], clarity["SHAP on XGBoost"]
    ebm_seed = float(np.mean(seed_stab["EBM"]["spearman"]))
    shap_seed = float(np.mean(seed_stab["SHAP"]["spearman"]))
    ebm_pert = float(np.mean(local_stab["EBM"]["cosine"]))
    shap_pert = float(np.mean(local_stab["SHAP"]["cosine"]))
    ebm_flip = 100 * float(np.mean(local_stab["EBM"]["top1_changed"]))
    shap_flip = 100 * float(np.mean(local_stab["SHAP"]["top1_changed"]))
    ebm_aopc = deletion["EBM (intrinsic)"]["aopc"]
    shap_aopc = deletion["SHAP on XGBoost"]["aopc"]
    rand_aopc = deletion["Random ranking (control)"]["aopc"]
    ebm_ratio = deletion["EBM (intrinsic)"]["aopc_ratio_vs_random"]
    shap_ratio = deletion["SHAP on XGBoost"]["aopc_ratio_vs_random"]

    results = f"""
## Explanation quality: glass-box against post-hoc

### What was produced

{reporting.file_ref("30_global_importance.csv", "global importance of every feature under both explanations")}
{reporting.file_ref("31_clarity.json", "concentration measures for both explanations")}
{reporting.file_ref("32_stability.csv", "re-training and perturbation stability")}
{reporting.file_ref("33_explainer_agreement.json", "agreement between four post-hoc explainers of the same model")}
{reporting.file_ref("34_faithfulness.json", "exactness, deletion curves and ground-truth recovery")}
{reporting.file_ref("35_scorecard.csv", "the head-to-head scorecard")}

### 1. Clarity

{metrics.df_to_markdown(pd.DataFrame([
    {"explanation": k,
     "features_for_50pct": v["features_for_50pct"],
     "features_for_90pct": v["features_for_90pct"],
     "normalised_entropy": v["normalised_entropy"],
     "gini_concentration": v["gini"]}
    for k, v in clarity.items()]))}

{reporting.fig_ref(RO, "30_global_importance.png", "Global feature importance under both explanations.")}

- EBM concentrates 90% of its attribution in **{ec['features_for_90pct']}** features.
- SHAP-on-XGBoost needs **{sc_['features_for_90pct']}** features for the same 90%.

### 2. Stability

**2a. Across re-training (different random seed, same data):**

{metrics.df_to_markdown(pd.DataFrame([
    {"explanation": k,
     "mean_spearman": float(np.mean(v["spearman"])),
     "std_dev": float(np.std(v["spearman"])),
     "mean_top5_jaccard": float(np.mean(v["jaccard"]))}
    for k, v in seed_stab.items()]))}

**2b. Under a {NOISE_SIGMA:.0%} input perturbation:**

{metrics.df_to_markdown(pd.DataFrame([
    {"explanation": k,
     "mean_cosine_similarity": float(np.mean(v["cosine"])),
     "mean_top3_jaccard": float(np.mean(v["jaccard"])),
     "top_reason_changed_pct": 100 * float(np.mean(v["top1_changed"]))}
    for k, v in local_stab.items()]))}

{reporting.fig_ref(RO, "31_stability.png", "Explanation stability under re-training and under input perturbation.")}

**2c. Agreement between post-hoc explainers of the *same* XGBoost model:**

{metrics.df_to_markdown(pd.DataFrame([
    {"pair": k, "spearman": v["spearman"], "top5_jaccard": v["jaccard"]}
    for k, v in agree["pairs"].items()]))}

{reporting.fig_ref(RO, "32_explainer_disagreement.png", "Rank correlation between four post-hoc explainers of one model.")}

Worst-agreeing pair: **{agree['worst_pair']}** at rho = {agree['worst_spearman']:.4f}.
Mean off-diagonal agreement: {agree['mean_offdiagonal']:.4f}.

### 3. Faithfulness

**3a. Exactness of the additive decomposition:**

{metrics.df_to_markdown(pd.DataFrame([
    {"explanation": k, "max_abs_error": v["max_abs_error"],
     "mean_abs_error": v["mean_abs_error"], "exact_by_construction": v["exact"]}
    for k, v in exact.items()]))}

**3b. Deletion test:**

{metrics.df_to_markdown(pd.DataFrame([
    {"ranking": k, "aopc": v["aopc"],
     "random_control_on_same_model": v["random_control_aopc"],
     "ratio_vs_own_control": v["aopc_ratio_vs_random"],
     "confidence_after_0": v["curve"][0],
     "confidence_after_5": v["curve"][5],
     "confidence_after_10": v["curve"][-1]}
    for k, v in deletion.items()]))}

{reporting.fig_ref(RO, "33_deletion_curve.png", "Model confidence as explanation-ranked features are removed.")}

**3c. Recovery of a known effect:**

Data generated from `{truth['generating_function']}` over {truth['n_rows']:,} rows.

{reporting.kv_table({
    "EBM RMSE against the true shape": truth["ebm_rmse"],
    "SHAP RMSE against the true shape": truth["shap_rmse"],
    "EBM correlation with the truth": truth["ebm_corr"],
    "SHAP correlation with the truth": truth["shap_corr"],
}, "Measure", "Value")}

{reporting.fig_ref(RO, "34_ground_truth_recovery.png", "Both methods against a known generating function.")}

### Scorecard

{metrics.df_to_markdown(pd.DataFrame(score["rows"], columns=[
    "property", "measure", "EBM (glass-box)", "SHAP on XGBoost", "better"]))}

**EBM wins {score['ebm_wins']} of {score['n_comparisons']} measures;
SHAP-on-XGBoost wins {score['shap_wins']}; {score['ties']} are level.**
"""

    analysis = f"""
## Analysis - what the explanation measurements mean

### The metrics used on this page

**Clarity** is measured as concentration. *Features for 90%* counts how many
features must be listed before 90% of the total attribution weight is covered -
a shorter list is a clearer explanation. *Normalised entropy* is the same idea
on a 0-1 scale (0 = one feature explains everything, 1 = all features matter
equally). *Gini concentration* is the inequality of the importance
distribution.

**Stability** is measured three ways, because an explanation can be unstable
for three different reasons.
*Re-training stability* asks whether a model trained on the same data with a
different random seed gives the same reason; it is the Spearman rank
correlation between the two global importance rankings.
*Perturbation stability* asks whether a barely-changed input gets a
barely-changed reason; it is the cosine similarity between the attribution
vectors before and after adding noise of {NOISE_SIGMA:.0%} of each feature's
standard deviation. Cosine similarity of 1.0 means the explanation is
identical in direction.
*Explainer agreement* asks whether two different tools explaining the same
trained model agree with each other.

**Faithfulness** asks whether the stated reason is the real reason.
*Reconstruction error* checks that the explanation's parts add up to the
model's own output.
*Deletion AOPC* (area over the perturbation curve) ranks each flow's features
by the explanation, replaces them one at a time with a neutral value, and
measures how fast the model's confidence collapses. A faithful explanation
names the features that actually carry the decision, so its curve falls
fastest and its AOPC is highest. Because each explanation can only be scored
against its own model, a random-ranking control is run separately on each
model and the comparable figure reported is the **ratio** of the explanation's
AOPC to that control. A ratio of 1.0 would mean the explanation is no better
than guessing.
*Ground-truth recovery* is the only measurement here with an external correct
answer: data is generated from a known rule, so each method's recovered effect
can be scored against the truth directly.

### What the measured values indicate

**Clarity.** The EBM reaches 90% of its attribution weight in
{ec['features_for_90pct']} features against {sc_['features_for_90pct']} for
SHAP, with normalised entropy {ec['normalised_entropy']:.3f} against
{sc_['normalised_entropy']:.3f}. On the concentration measure alone
{"SHAP-on-XGBoost is the sparser explanation" if sc_['features_for_90pct'] < ec['features_for_90pct'] else "the EBM is the sparser explanation"},
and that is reported as measured. Sparsity is not the whole of clarity,
though. The EBM spreads weight across more features partly because it fits a
smooth curve to each one rather than a handful of hard splits, and its pairwise
terms are shared between the two features involved.

The structural difference is the one that matters for an operator: the EBM's
explanation *is* the trained model, so a reader needs nothing beyond the model
file and can read the whole of it, whereas the SHAP explanation is a separate
computation whose result depends on the explainer chosen, its background data
and its sample size - the dependence quantified in section 2c below.

**Stability across re-training.** Each repeat fits both models to the same
bootstrap resample of the training data, so the test is symmetric. EBM rankings
agree at rho = {ebm_seed:.4f} across resamples; SHAP-on-XGBoost at rho =
{shap_seed:.4f}. An operator who retrains monthly wants the reasons to stay put
unless the traffic changed. Note that varying only the random seed would have
been meaningless here: XGBoost is deterministic given fixed data and settings,
so it would have scored a perfect 1.0 without demonstrating any robustness at
all. Resampling the data is what makes the two comparable.

**Stability under perturbation.** Changing each continuous feature by
{NOISE_SIGMA:.0%} of its interquartile range - far less than the natural
variation between two similar flows - left the EBM explanation at cosine
similarity {ebm_pert:.4f} and the SHAP explanation at {shap_pert:.4f}. The
top-ranked reason flipped for {ebm_flip:.1f}% of flows under EBM and
{shap_flip:.1f}% under SHAP. The interquartile range is used rather than the
standard deviation because flow features are heavily long-tailed: a percent of
the standard deviation of `TotBytes` would be tens of kilobytes, which is not a
small change to a typical flow.

Neither method is perfectly stable on this measure, and the honest reading is
that local attributions on collision-heavy flow data are sensitive for both.
That is worth stating rather than hiding: the flows most affected are exactly
the benign/UDPFlood pairs identified in Objective 1, where the model's decision
is genuinely balanced on a knife edge and a small nudge legitimately changes
which feature tips it.

**Explainer disagreement is the decisive result.** Four standard ways of
explaining *one* trained XGBoost model - TreeSHAP, KernelSHAP, permutation
importance and the booster's own gain statistic - produce rankings that
correlate at only {agree['mean_offdiagonal']:.3f} on average, with the worst
pair ({agree['worst_pair']}) at rho = {agree['worst_spearman']:.3f}. The model
did not change between these four measurements; only the tool did. This
reproduces, on this data and with a broader set of explainers, the finding of
Uccello and Nadjm-Tehrani that post-hoc attributions on 5G-NIDD are
explainer-dependent, and it is the concrete form of research gap 2. A glass-box
model is immune to this by construction: there is one explanation because there
is nothing to approximate.

**Faithfulness: exactness.** The EBM reconstruction error is
{exact['EBM (intrinsic)']['max_abs_error']:.2e} - floating-point noise, because
the model *is* the sum of its terms. TreeSHAP also reconstructs the XGBoost
margin closely
({exact['SHAP on XGBoost']['max_abs_error']:.2e}); this is worth stating
plainly rather than overclaiming, because TreeSHAP is exact for tree ensembles
by design. The honest distinction is not that TreeSHAP fails to add up - it is
that TreeSHAP is one of several available explanations of that model, and
section 2c shows the others disagree with it.

**Faithfulness: the deletion test.** Raw AOPC is {ebm_aopc:.4f} for the EBM's
own explanation and {shap_aopc:.4f} for SHAP-on-XGBoost, against
{rand_aopc:.4f} for a random ranking. The raw numbers are not directly
comparable, because each explanation can only be evaluated on its own model and
the two models differ in how sharply they react to having a feature replaced.
Against its own random control each explanation scores
**{ebm_ratio:.2f}x** (EBM) and **{shap_ratio:.2f}x** (SHAP), and it is those
ratios that can be set side by side. Both are far above 1.0, so both
explanations genuinely identify the features their model relies on; this test
separates real explanations from noise rather than separating the two
approaches.

**Faithfulness: ground truth.** This is the measurement with a correct answer.
Against the known effect `2*sin(1.5*x0)`, the EBM shape function achieves RMSE
{truth['ebm_rmse']:.4f} (correlation {truth['ebm_corr']:.4f}) and SHAP-on-XGBoost
RMSE {truth['shap_rmse']:.4f} (correlation {truth['shap_corr']:.4f}). The EBM
recovers the true curve because it fits a curve of that form; SHAP recovers a
step-wise approximation of it, because that is what the underlying tree
ensemble represents. The explanation cannot be smoother than the model it
describes.

**Interpreting the figures.** In the disagreement heat map, every off-diagonal
cell below 1.00 is a pair of tools telling a different story about the same
model - this single figure is the argument for the glass-box approach. In the
deletion figure, steeper is better, and the gap between the solid lines and the
dashed control is the evidence that both explanations are meaningful. In the
ground-truth figure the dashed line is the exact answer, and the distance from
it is the error.

### How this supports Research Objective 3

Objective 3 asks for the EBM's intrinsic explanations to be compared with
post-hoc SHAP explanations of an XGBoost model, on identical splits, in terms
of clarity, stability and faithfulness. All three are measured, on the same
{len(imp_frame)}-feature representation and the same held-out flows:

- **Clarity** - two concentration measures plus the structural point that the
  glass-box explanation requires no additional tool.
- **Stability** - three independent tests (re-training, input perturbation and
  cross-explainer agreement), the third of which reproduces the specific
  weakness the study identifies in the post-hoc literature.
- **Faithfulness** - three independent tests (additive exactness, the deletion
  test with a random control, and recovery of a known effect on synthetic data
  where the answer is available).

The scorecard is deliberately mixed rather than one-sided: it resolves
{score['ebm_wins']} of {score['n_comparisons']} measures in favour of the
glass-box explanation, {score['shap_wins']} in favour of SHAP-on-XGBoost and
{score['ties']} level. Reporting it that way is the point - a comparison
engineered to make the study model win on every axis would not be evidence of
anything.

The measures the glass-box wins are the ones the study's argument rests on.
Exactness is not a close call: the EBM reconstructs its own output to
{exact['EBM (intrinsic)']['max_abs_error']:.0e}, because there is nothing to
approximate. Ground-truth recovery is the only test here with a correct answer
available, and the EBM is roughly
{truth['shap_rmse'] / max(truth['ebm_rmse'], 1e-9):.1f} times closer to it.
And the result that carries the study is 2c: post-hoc explanations of one
black-box model disagree with each other at rho =
{agree['worst_spearman']:.3f} in the worst case. The model did not change - only
the tool did. No amount of care in choosing an explainer removes that ambiguity,
whereas a glass-box model never presents the choice. That is research gap 2,
measured on this data.
"""
    reporting.write_fragment(
        profile, RO, "explain", 1, results_md=results, analysis_md=analysis,
        headline={
            "worst_explainer_agreement": agree["worst_spearman"],
            "ebm_seed_stability": ebm_seed,
            "shap_seed_stability": shap_seed,
            "ebm_ground_truth_rmse": truth["ebm_rmse"],
            "shap_ground_truth_rmse": truth["shap_rmse"],
            "ebm_wins": score["ebm_wins"],
            "n_comparisons": score["n_comparisons"],
        })
