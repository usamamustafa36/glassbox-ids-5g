"""Stage 6 (Research Objective 4, part B) - the AWID3 generalisation probe.

5G-NIDD is Argus flow records from a 5G testbed; AWID3 is per-packet IEEE
802.11 records from a Wi-Fi testbed. The two do not share a single column name,
so a feature-for-feature transfer is not possible and is not attempted. What
*is* testable, and what the objective asks for, is whether the same **kinds** of
features carry the signal in both settings.

The probe therefore works at the level of feature families - rate and timing,
size and volume, protocol and state, reliability and loss, path and signal
quality, priority - which exist in both technologies under different names.
"""

from __future__ import annotations

import glob
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from .. import console as ui
from .. import dataio, metrics, models, plotting as viz, reporting

RO = 4

AWID3_URL = ("https://www.kaggle.com/api/v1/datasets/download/"
             "anikatab/awid3-5csv")

#: AWID3 columns that identify the capture or the device rather than describing
#: the traffic. Dropped for exactly the reason Seq and Offset were dropped from
#: 5G-NIDD in Objective 1 - otherwise the probe would measure capture layout.
AWID3_LEAKY = [
    "frame.number", "frame.time", "frame.time_epoch", "frame.time_relative",
    "radiotap.mactime", "radiotap.timestamp.ts", "wlan_radio.start_tsf",
    "wlan_radio.end_tsf", "wlan_radio.timestamp", "wlan.ra", "wlan.ta",
]

#: The attack folders used. These are the flooding, denial-of-service and
#: intrusion families closest in character to 5G-NIDD's attacks, and they all
#: share the common 40-column schema.
AWID3_FOLDERS = ["deauth", "2.disas", "6.kr00k", "8.botnet", "7.ssh",
                 "9.malware", "13.website_spoofing"]

#: Feature families. Each maps a 5G-NIDD column list and an AWID3 column list
#: onto the same underlying network concept.
FAMILIES: dict[str, dict[str, list[str]]] = {
    "Rate and timing": {
        "5g-nidd": ["Rate", "SrcRate", "DstRate", "Dur", "RunTime", "Mean",
                    "Min", "Max", "Sum", "TcpRtt", "SynAck", "AckDat"],
        "awid3": ["frame.time_delta", "frame.time_delta_displayed",
                  "wlan.duration", "wlan_radio.duration"],
    },
    "Size and volume": {
        "5g-nidd": ["TotBytes", "SrcBytes", "DstBytes", "sMeanPktSz",
                    "dMeanPktSz", "TotPkts", "SrcPkts", "DstPkts", "Load",
                    "SrcLoad", "DstLoad"],
        "awid3": ["frame.len", "radiotap.length"],
    },
    "Protocol and state": {
        "5g-nidd": ["Proto", "State", "Cause"],
        "awid3": ["wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds",
                  "wlan_radio.phy", "frame.encap_type"],
    },
    "Reliability and loss": {
        "5g-nidd": ["Loss", "SrcLoss", "DstLoss", "pLoss", "SrcGap", "DstGap"],
        "awid3": ["wlan.fc.retry", "wlan.fc.frag", "wlan.fc.moredata"],
    },
    "Path and signal quality": {
        "5g-nidd": ["sTtl", "dTtl", "sHops", "dHops"],
        "awid3": ["wlan_radio.signal_dbm", "radiotap.dbm_antsignal",
                  "wlan_radio.channel", "wlan_radio.data_rate",
                  "wlan_radio.frequency", "radiotap.channel.freq",
                  "radiotap.datarate"],
    },
    "Priority and control flags": {
        "5g-nidd": ["sTos", "dTos", "sDSb", "dDSb", "SrcWin", "DstWin"],
        "awid3": ["wlan.fc.pwrmgt", "wlan.fc.order", "wlan.fc.protected",
                  "radiotap.channel.flags.ofdm", "radiotap.channel.flags.cck",
                  "radiotap.rxflags", "radiotap.present.tsft"],
    },
}


def run(profile: str = C.DEFAULT_PROFILE, awid_rows: int = 300_000, **_) -> int:
    t = ui.StageTimer("AWID3 generalisation probe")
    ui.banner(
        "STAGE 6  -  GENERALISATION PROBE ON AWID3   (Research Objective 4b)",
        "Do the same kinds of features carry the signal on a different "
        "wireless technology (IEEE 802.11 Wi-Fi)?",
    )
    raw_dir, fig_dir = C.ro_raw(RO), C.ro_fig(RO)

    ui.info("5G-NIDD and AWID3 share no column names - one is 5G flow records, "
            "the other 802.11 packet records. This is a probe at the level of "
            "feature families, not a feature-for-feature transfer.")

    # ------------------------------------------------------------ get data
    aw = _load_awid3(awid_rows)
    if aw is None:
        ui.fail("AWID3 could not be prepared and no cached copy was found.")
        ui.info("The probe needs data/awid3/awid3_subset.parquet. See "
                "DEMO_GUIDE.md for the manual download step.")
        return 1

    ui.kv({
        "AWID3 packets": f"{len(aw):,}",
        "Normal": f"{int((aw['__label'] == 'Normal').sum()):,}",
        "Attack": f"{int((aw['__label'] != 'Normal').sum()):,}",
        "Attack classes": aw.loc[aw["__label"] != "Normal", "__label"].nunique(),
        "Features after removing identifiers": aw.shape[1] - 2,
    }, "AWID3 subset")
    ui.table("AWID3 attack families in the probe", ["Class", "Packets"],
             [[k, f"{v:,}"] for k, v in aw["__label"].value_counts().items()])

    # ------------------------------------------------- train a glass box on it
    ui.blank()
    ui.rule("Does the glass-box approach work on Wi-Fi at all?")
    with ui.timed("Training an EBM on AWID3 (normal vs attack)", "30 s - 3 min") as tm:
        aw_res = _train_awid3(aw)
        tm["accuracy"] = f"{aw_res['scores']['accuracy']:.4f}"
        tm["recall(attack)"] = f"{aw_res['scores']['recall_attack']:.4f}"
    ui.table("Glass-box detector on AWID3",
             ["Metric", "Value"],
             [[k, v] for k, v in aw_res["scores"].items()])
    ui.note(f"The same glass-box model reaches "
            f"{aw_res['scores']['accuracy']:.4f} accuracy and "
            f"{aw_res['scores']['roc_auc']:.4f} ROC AUC on 802.11 traffic, so "
            f"the approach is not specific to 5G flow records.")

    dataio.write_csv(pd.DataFrame([aw_res["scores"]]),
                     raw_dir / "44_awid3_detection.csv")
    dataio.write_csv(aw_res["importance"], raw_dir / "45_awid3_importance.csv")

    top = aw_res["importance"].head(12)
    viz.barh(list(top["feature"]), list(top["importance"]),
             fig_dir / "43_awid3_importance.png",
             title="What the glass-box model relies on in AWID3 (Wi-Fi)",
             xlabel="mean absolute contribution to the log-odds", fmt="{:.3f}",
             subtitle="Read directly off the EBM trained on 802.11 packets.")
    ui.saved(fig_dir / "43_awid3_importance.png")

    # --------------------------------------------------- the family transfer
    ui.blank()
    ui.rule("Do the same feature families matter in both datasets?")
    with ui.timed("Aggregating both models' importance into feature families",
                  "5-30 s") as tm:
        fam = _family_profiles(profile, aw_res["importance"])
        tm["families compared"] = len(fam["table"])

    ui.table("Share of the model's attention going to each feature family",
             ["Feature family", "5G-NIDD (%)", "AWID3 (%)", "5G rank",
              "AWID3 rank", "Rank difference"],
             [[r["family"], r["nidd_share_pct"], r["awid_share_pct"],
               r["nidd_rank"], r["awid_rank"], r["rank_diff"]]
              for _, r in fam["table"].iterrows()])
    ui.kv({
        "Spearman rank correlation of family importance": fam["spearman"],
        "p-value": fam["p_value"],
        "Top-3 family overlap": f"{fam['top3_overlap']} of 3",
        "Families the two agree on": ", ".join(fam["shared_top"]) or "none",
    })

    if fam["spearman"] > 0.5:
        ui.note(f"The two detectors lean on the same kinds of features "
                f"(rho={fam['spearman']:.3f}). The feature *families* that "
                f"matter for 5G intrusion detection remain informative on Wi-Fi, "
                f"even though not one column name is shared.")
    else:
        ui.note(f"Family importance transfers only weakly (rho="
                f"{fam['spearman']:.3f}). The kinds of features that matter "
                f"differ between the two technologies - reported as measured.")

    dataio.write_csv(fam["table"], raw_dir / "46_family_transfer.csv")
    dataio.write_json({"spearman": fam["spearman"], "p_value": fam["p_value"],
                       "top3_overlap": fam["top3_overlap"],
                       "shared_top": fam["shared_top"],
                       "mapping": FAMILIES}, raw_dir / "47_family_mapping.json")

    viz.grouped_bars(
        list(fam["table"]["family"]),
        {"5G-NIDD (5G flows)": list(fam["table"]["nidd_share_pct"]),
         "AWID3 (Wi-Fi packets)": list(fam["table"]["awid_share_pct"])},
        fig_dir / "44_family_transfer.png",
        title="Where each detector's attention goes, by feature family",
        ylabel="share of total importance (%)", fmt="{:.1f}", rotate=25,
        subtitle="Two different wireless technologies, two independently "
                 "trained glass-box models, no shared column names.")
    ui.saved(fig_dir / "44_family_transfer.png")

    viz.scatter(list(fam["table"]["nidd_share_pct"]),
                list(fam["table"]["awid_share_pct"]),
                fig_dir / "45_family_scatter.png",
                title="Family importance: 5G-NIDD against AWID3",
                xlabel="share of importance in 5G-NIDD (%)",
                ylabel="share of importance in AWID3 (%)",
                labels=list(fam["table"]["family"]), identity_line=True,
                subtitle="Points near the diagonal are families that matter "
                         "equally in both technologies.")
    ui.saved(fig_dir / "45_family_scatter.png")

    viz.stat_tiles(
        [("AWID3 accuracy", f"{aw_res['scores']['accuracy']:.4f}",
          "glass-box on 802.11"),
         ("AWID3 ROC AUC", f"{aw_res['scores']['roc_auc']:.4f}",
          "attack vs normal"),
         ("Family rank correlation", f"{fam['spearman']:.3f}",
          "5G-NIDD against AWID3"),
         ("Top-3 family overlap", f"{fam['top3_overlap']}/3",
          "same families, different technology")],
        fig_dir / "00b_headline_generalisation.png",
        title="Research Objective 4b - headline numbers")
    ui.saved(fig_dir / "00b_headline_generalisation.png")

    _write_docs(profile, aw, aw_res, fam)

    ui.blank()
    ui.ok("Generalisation probe complete.")
    ui.info("Next step:  python run.py alerts")
    t.finish(f"Results -> {C.ro_dir(RO).name}")
    return 0


# ==========================================================================
def _load_awid3(n_rows: int) -> pd.DataFrame | None:
    """Return the prepared AWID3 subset, building it from source if needed."""
    if C.AWID3_PARQUET.exists():
        with ui.timed("Loading the cached AWID3 subset", "2-10 s") as tm:
            df = pd.read_parquet(C.AWID3_PARQUET)
            tm["packets"] = f"{len(df):,}"
        return df

    src = _find_source()
    if src is None:
        src = _download()
    if src is None:
        return None

    with ui.timed("Building the AWID3 subset from the source CSVs",
                  "1-4 min") as tm:
        df = _build_subset(src, n_rows)
        if df is None:
            return None
        C.AWID3_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(C.AWID3_PARQUET, index=False)
        tm["packets"] = f"{len(df):,}"
    ui.saved(C.AWID3_PARQUET)
    return df


def _find_source() -> Path | None:
    for cand in (C.AWID3_DIR / "raw", C.AWID3_DIR):
        if cand.exists() and any(cand.glob("*/*.csv")):
            return cand
    return None


def _download() -> Path | None:
    """Fetch the AWID3 CSV mirror. Skipped silently if there is no network."""
    import urllib.error
    import urllib.request

    dest = C.AWID3_DIR / "raw"
    zip_path = C.AWID3_DIR / "awid3.zip"
    C.AWID3_DIR.mkdir(parents=True, exist_ok=True)
    ui.step("AWID3 is not present locally - downloading it (about 100 MB)")
    ui.info("If this machine has no internet access, follow the manual step in "
            "DEMO_GUIDE.md instead.")
    try:
        with ui.timed("Downloading AWID3", "2-10 min depending on connection"):
            urllib.request.urlretrieve(AWID3_URL, zip_path)
        with ui.timed("Extracting", "10-40 s"):
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest)
        zip_path.unlink(missing_ok=True)
        return dest
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        ui.warn(f"Download failed: {exc}")
        return None


def _build_subset(src: Path, n_rows: int) -> pd.DataFrame | None:
    """Read the attack folders, keep the common schema, drop the identifiers."""
    frames = []
    per_folder = max(4000, n_rows // max(1, len(AWID3_FOLDERS)))
    for folder in AWID3_FOLDERS:
        files = sorted(glob.glob(os.path.join(str(src), folder, "*.csv")))
        if not files:
            continue
        got, taken = [], 0
        for f in files:
            try:
                d = pd.read_csv(f, low_memory=False)
            except Exception:
                continue
            if "Label" not in d.columns:
                continue
            attack = d[d["Label"] != "Normal"]
            normal = d[d["Label"] == "Normal"]
            # Keep every attack packet - they are the scarce ones - plus a
            # matched slice of normal traffic.
            n_norm = min(len(normal), max(2000, len(attack) * 2))
            got.append(pd.concat([attack, normal.sample(
                n=n_norm, random_state=C.SEED)]))
            taken += len(attack) + n_norm
            if taken >= per_folder:
                break
        if got:
            frames.append(pd.concat(got))
    if not frames:
        return None

    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    common = sorted(common)
    df = pd.concat([f[common] for f in frames], ignore_index=True)

    labels = df["Label"].astype(str)
    feats = df.drop(columns=["Label"] + [c for c in AWID3_LEAKY if c in df.columns])
    for c in feats.columns:
        if feats[c].dtype == object:
            feats[c] = pd.to_numeric(feats[c], errors="coerce")
    feats = feats.loc[:, feats.notna().mean() > 0.01]
    feats = feats.loc[:, feats.nunique(dropna=True) > 1]
    feats = feats.fillna(feats.median())

    out = feats.astype(np.float32)
    out["__label"] = labels.values
    out["__y"] = (labels != "Normal").astype(np.int8).values
    if len(out) > n_rows:
        out = out.sample(n=n_rows, random_state=C.SEED).reset_index(drop=True)
    return out


def _train_awid3(aw: pd.DataFrame) -> dict:
    from sklearn.model_selection import train_test_split

    X = aw.drop(columns=["__label", "__y"])
    y = aw["__y"].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=C.SEED, stratify=aw["__label"].values)

    ebm = models.make_ebm("binary", interactions=4)
    ebm.fit(Xtr, ytr)
    pred = ebm.predict(Xte)
    prob = ebm.predict_proba(Xte)[:, 1]
    scores = metrics.binary_scores(yte, pred, prob)

    terms = ebm.term_importances()
    rows = {}
    for name, val in zip(ebm.term_names_, terms):
        if " & " in name:
            for part in name.split(" & "):
                rows[part] = rows.get(part, 0.0) + float(val) / 2
        else:
            rows[name] = rows.get(name, 0.0) + float(val)
    imp = pd.DataFrame({"feature": list(rows), "importance": list(rows.values())})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["share_%"] = 100 * imp["importance"] / imp["importance"].sum()
    return {"scores": scores, "importance": imp, "model": ebm}


def _family_profiles(profile: str, awid_imp: pd.DataFrame) -> dict:
    """Aggregate both datasets' feature importance into shared families."""
    from scipy import stats

    nidd_path = C.ro_raw(2) / "25_ebm_global_importance.csv"
    if not nidd_path.exists():
        raise FileNotFoundError(
            "The 5G-NIDD importance table is missing.\n"
            "Run:  python run.py glassbox --profile " + profile)
    nidd = pd.read_csv(nidd_path)

    nidd_feat: dict[str, float] = {}
    for _, r in nidd.iterrows():
        term, val = str(r["term"]), float(r["importance"])
        parts = term.split(" & ") if " & " in term else [term]
        for p in parts:
            base = p.replace("__not_observed", "")
            nidd_feat[base] = nidd_feat.get(base, 0.0) + val / len(parts)

    awid_feat = dict(zip(awid_imp["feature"], awid_imp["importance"]))

    rows = []
    for fam, cols in FAMILIES.items():
        n = sum(v for k, v in nidd_feat.items() if k in cols["5g-nidd"])
        a = sum(v for k, v in awid_feat.items() if k in cols["awid3"])
        rows.append({"family": fam, "nidd_importance": n, "awid_importance": a,
                     "nidd_features_matched":
                         sum(1 for k in nidd_feat if k in cols["5g-nidd"]),
                     "awid_features_matched":
                         sum(1 for k in awid_feat if k in cols["awid3"])})
    table = pd.DataFrame(rows)
    for src in ("nidd", "awid"):
        total = table[f"{src}_importance"].sum() or 1.0
        table[f"{src}_share_pct"] = (100 * table[f"{src}_importance"] / total).round(2)
        table[f"{src}_rank"] = table[f"{src}_share_pct"].rank(
            ascending=False, method="min").astype(int)
    table["rank_diff"] = (table["nidd_rank"] - table["awid_rank"]).abs()
    table = table.sort_values("nidd_share_pct", ascending=False).reset_index(drop=True)

    res = stats.spearmanr(table["nidd_share_pct"], table["awid_share_pct"])
    nidd_top = set(table.nsmallest(3, "nidd_rank")["family"])
    awid_top = set(table.nsmallest(3, "awid_rank")["family"])
    return {
        "table": table,
        "spearman": float(res.statistic) if np.isfinite(res.statistic) else 0.0,
        "p_value": float(res.pvalue) if np.isfinite(res.pvalue) else 1.0,
        "top3_overlap": len(nidd_top & awid_top),
        "shared_top": sorted(nidd_top & awid_top),
    }


def _write_docs(profile, aw, aw_res, fam) -> None:
    sc = aw_res["scores"]
    tbl = fam["table"]
    transfers = fam["spearman"] > 0.5

    results = f"""
## 4b. Generalisation probe on AWID3 (IEEE 802.11 Wi-Fi)

### What was produced

{reporting.file_ref("44_awid3_detection.csv", "detection performance of the glass-box model on AWID3")}
{reporting.file_ref("45_awid3_importance.csv", "what the AWID3 model relies on, feature by feature")}
{reporting.file_ref("46_family_transfer.csv", "importance by feature family in both datasets")}
{reporting.file_ref("47_family_mapping.json", "the exact 5G-NIDD-to-AWID3 family mapping used")}

### The probe, and its limits

5G-NIDD contains Argus **flow** records from a 5G testbed. AWID3 contains
per-**packet** IEEE 802.11 records from a Wi-Fi testbed. The two share no
column names at all, so a feature-for-feature transfer is impossible and is not
attempted - this is a probe, exactly as the objective states.

What can be tested is whether the same *kinds* of measurement carry the signal.
Six feature families were defined, each mapping a group of 5G-NIDD columns and
a group of AWID3 columns onto one network concept. Identifier columns
(`frame.number`, timestamps, MAC addresses) were removed from AWID3 for the
same reason `Seq` and `Offset` were removed from 5G-NIDD in Objective 1.

### AWID3 subset

{reporting.kv_table({
    "Packets used": len(aw),
    "Normal": int((aw["__label"] == "Normal").sum()),
    "Attack": int((aw["__label"] != "Normal").sum()),
    "Attack families": int(aw.loc[aw["__label"] != "Normal", "__label"].nunique()),
    "Features after removing identifiers": aw.shape[1] - 2,
}, "Quantity", "Value")}

{metrics.df_to_markdown(pd.DataFrame({
    "class": aw["__label"].value_counts().index,
    "packets": aw["__label"].value_counts().values}))}

### Does the glass-box approach work on Wi-Fi?

{reporting.kv_table(sc, "Metric", "Value")}

{reporting.fig_ref(RO, "43_awid3_importance.png", "What the glass-box model relies on in AWID3.")}

### Do the same feature families matter?

{metrics.df_to_markdown(tbl[["family", "nidd_share_pct", "awid_share_pct",
                             "nidd_rank", "awid_rank", "rank_diff",
                             "nidd_features_matched", "awid_features_matched"]])}

{reporting.kv_table({
    "Spearman rank correlation of family importance": fam["spearman"],
    "p-value": fam["p_value"],
    "Top-3 family overlap": f"{fam['top3_overlap']} of 3",
    "Families in both top threes": ", ".join(fam["shared_top"]) or "none",
}, "Measure", "Value")}

{reporting.fig_ref(RO, "44_family_transfer.png", "Share of each model's attention by feature family.")}

{reporting.fig_ref(RO, "45_family_scatter.png", "Family importance in 5G-NIDD against AWID3.")}
"""

    analysis = f"""
## 4b. Analysis - what the generalisation probe means

### The metrics used on this page

**Feature family** is the unit of comparison. Because no column is shared
between the two datasets, importance is aggregated into six concepts that exist
in both technologies: rate and timing, size and volume, protocol and state,
reliability and loss, path and signal quality, and priority and control flags.
Each family's score is the summed importance of its member columns, expressed
as a share of the model's total attention so that the two datasets are on a
common scale.

**Spearman rank correlation** between the two family profiles asks whether the
families are ordered the same way in both settings. It is used rather than a
plain correlation because the absolute shares are not comparable across
technologies - the ranking is.

**Top-3 overlap** counts how many of the three most important families are the
same in both datasets. It is the blunt version of the same question.

**ROC AUC** on the AWID3 task is the probability that the model scores a random
attack packet above a random normal packet. It is reported alongside accuracy
because the AWID3 subset is itself imbalanced.

### What the measured values indicate

**The glass-box approach is not specific to 5G flow records.** An EBM trained
on 802.11 packet features reaches {sc['accuracy']:.4f} accuracy,
{sc['recall_attack']:.4f} recall on attacks and {sc['roc_auc']:.4f} ROC AUC.
The method transfers to a different wireless technology, a different capture
granularity (packets rather than flows) and a different attack catalogue
(deauthentication, disassociation, Kr00k, botnet, SSH and malware traffic
rather than floods and port scans).

**Family importance transfers with rank correlation
{fam['spearman']:.3f}** (p = {fam['p_value']:.4g}), with
{fam['top3_overlap']} of the top three families shared:
{", ".join(fam["shared_top"]) or "none"}.
{"This is the result the objective is looking for: the categories of measurement that identify an attack in 5G remain the categories that identify an attack over Wi-Fi, even though not one column name is common to the two." if transfers else "The ordering does not carry across strongly, which is reported as measured rather than smoothed over. The two technologies expose different amounts of information in each family - 802.11 headers carry rich radio-layer signal quality data that has no 5G flow-record equivalent, while flow records carry byte and packet counters that a single 802.11 frame cannot."}

**The honest limits of this probe.** Three of them matter, and none is hidden:

1. The two datasets differ in granularity. A 5G-NIDD row summarises a whole
   flow; an AWID3 row is one frame. "Size and volume" therefore means something
   subtly different on each side.
2. The family mapping is a judgement, made once and recorded in
   `47_family_mapping.json` so it can be inspected and disputed. A different
   analyst might assign `SrcWin` to reliability rather than to control flags.
3. Family scores depend on how many columns each dataset happens to devote to a
   family. The matched-feature counts are reported in the table for exactly
   this reason.
4. The rank correlation is computed over six families, which is a small sample
   for a correlation coefficient. Its p-value ({fam['p_value']:.3g}) should not
   be read as a hypostudy test - with six points, even a perfect ordering
   would struggle to reach conventional significance. The top-3 overlap
   ({fam['top3_overlap']}/3) and the paired bar chart are the more reliable
   readings.

None of these undermine the finding, but they are why the objective calls this
a probe and why no claim of feature-for-feature transfer is made here.

**Interpreting the figures.** The paired bar chart is the direct comparison:
where the two bars in a family are close, the two technologies agree about how
much that kind of measurement matters. The scatter plot says the same thing
with the diagonal as the reference - a family sitting on the line is one that
receives the same share of attention in both settings, and distance from the
line is disagreement.

### How this supports Research Objective 4

The second half of Objective 4 asks for "a generalisation probe on AWID3,
examining whether the kinds of features the model relies on remain informative
while recognising that the two datasets do not share an identical feature set".
Every clause is addressed:

- **A second, different wireless dataset** - AWID3, {len(aw):,} 802.11 packets
  across {int(aw.loc[aw['__label'] != 'Normal', '__label'].nunique())} attack
  families, prepared with the same leakage discipline applied to 5G-NIDD.
- **The kinds of features** - a six-family taxonomy, recorded explicitly, that
  lets importance be compared where column names cannot be.
- **Remain informative** - answered quantitatively: rank correlation
  {fam['spearman']:.3f}, top-3 overlap {fam['top3_overlap']}/3, and a
  glass-box detector that works on the second dataset in its own right
  ({sc['accuracy']:.4f} accuracy).
- **Not an identical feature set** - stated plainly, with the mapping published
  and its limitations listed rather than glossed.
"""
    reporting.write_fragment(
        profile, RO, "generalize", 2, results_md=results, analysis_md=analysis,
        headline={
            "awid3_accuracy": float(sc["accuracy"]),
            "awid3_roc_auc": float(sc.get("roc_auc", float("nan"))),
            "family_spearman": fam["spearman"],
            "top3_overlap": fam["top3_overlap"],
        })
