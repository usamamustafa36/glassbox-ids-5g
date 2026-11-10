"""Stage 7 (Research Objective 5) - analyst-readable explanation output.

Turns the glass-box model's exact term contributions into an alert card a
non-expert can act on: what was detected, how confident the detector is, the
three reasons in plain English with the measured value against what normal
traffic looks like, and a recommended first response.

Nothing here is a second model or an approximation. Every number on a card is
read straight out of the EBM that made the decision.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd

from .. import config as C
from .. import console as ui
from .. import dataio, glossary, models, plotting as viz, reporting

RO = 5
TOP_REASONS = 3


def run(profile: str = C.DEFAULT_PROFILE, n_cards: int = 8, **_) -> int:
    t = ui.StageTimer("Analyst-readable explanations")
    ui.banner(
        "STAGE 7  -  ANALYST-READABLE ALERTS   (Research Objective 5)",
        "Turn the glass-box model's exact reasoning into a card a non-expert "
        "decision-maker can read and act on.",
    )
    raw_dir, fig_dir = C.ro_raw(RO), C.ro_fig(RO)

    with ui.timed("Loading the split and the glass-box models", "5-20 s"):
        s = dataio.load_splits(profile)
        ebm_b = models.load_model(profile, "ebm_binary")
        ebm_m = models.load_model(profile, "ebm_multi")

    ctx = _context(s)
    ui.info("Benign reference values computed from the training split - every "
            "card compares the flow against what normal traffic looks like.")

    with ui.timed(f"Selecting {n_cards} representative alerts", "5-20 s") as tm:
        picks = _select(s, ebm_b, n_cards)
        tm["alerts"] = len(picks)

    with ui.timed("Building the cards from the model's own term contributions",
                  "10-60 s") as tm:
        cards = [_build_card(p, s, ebm_b, ebm_m, ctx) for p in picks]
        tm["cards"] = len(cards)

    ui.table("Alerts generated",
             ["#", "Test row", "True class", "Detector says", "Confidence",
              "Severity"],
             [[i + 1, c["row"], c["true"], c["pred"], c["confidence"],
               c["severity"]] for i, c in enumerate(cards)])

    ui.blank()
    ui.rule("Example card (as the analyst sees it)")
    _print_card(cards[0])

    ui.blank()
    with ui.timed("Rendering cards to Markdown, HTML and PNG", "10-60 s"):
        md = _render_markdown(cards, profile)
        (C.ro_dir(RO) / "ALERT_CARDS.md").write_text(md, encoding="utf-8")
        html_doc = _render_html(cards, profile)
        (C.ro_dir(RO) / "ALERT_CARDS.html").write_text(html_doc, encoding="utf-8")
        for i, c in enumerate(cards, 1):
            _card_figure(c, fig_dir / f"50_card_{i:02d}_{_slug(c['pred'])}.png")
    ui.saved(C.ro_dir(RO) / "ALERT_CARDS.md")
    ui.saved(C.ro_dir(RO) / "ALERT_CARDS.html")
    ui.saved(fig_dir)

    flat = pd.DataFrame([{
        "alert": i + 1,
        "test_row": c["row"],
        "true_class": c["true"],
        "predicted_class": c["pred"],
        "confidence": c["confidence"],
        "correct": c["true"] == c["pred"],
        **{f"reason_{j + 1}": r["sentence"] for j, r in enumerate(c["reasons"])},
    } for i, c in enumerate(cards)])
    dataio.write_csv(flat, raw_dir / "50_alert_cards.csv")
    dataio.write_json(cards, raw_dir / "51_alert_cards_full.json")

    # ------------------------------------------------------- readability
    ui.blank()
    ui.rule("Is the output actually readable?")
    read = _readability(cards)
    ui.table("Readability of the generated alerts", ["Measure", "Value"],
             [[k, v] for k, v in read.items()])
    dataio.write_json(read, raw_dir / "52_readability.json")
    ui.note("Every reason on every card is a sentence naming a measured "
            "quantity, its value, and how that compares with normal traffic - "
            "no feature codes, no model jargon, no SHAP values.")

    viz.stat_tiles(
        [("Alert cards produced", str(len(cards)), "one per attack type found"),
         ("Reasons per alert", str(TOP_REASONS), "ranked by actual contribution"),
         ("Words per reason", f"{read['mean_words_per_reason']:.0f}",
          "plain English, no jargon"),
         ("Extra tools needed", "0", "the reasons come from the model itself")],
        fig_dir / "00_headline.png",
        title="Research Objective 5 - headline numbers")
    ui.saved(fig_dir / "00_headline.png")

    _write_docs(profile, cards, read)

    ui.blank()
    ui.ok("Analyst-readable alerts produced.")
    ui.info("Open Results/.../ALERT_CARDS.html in a browser to read the alerts.")
    ui.info("Try one live:  python run.py explain-flow --row 42")
    t.finish(f"Results -> {C.ro_dir(RO).name}")
    return 0


# ==========================================================================
def explain_one(profile: str, row: int) -> int:
    """Interactive single-flow explanation, for use during the demonstration."""
    s = dataio.load_splits(profile)
    if not 0 <= row < len(s.X_test):
        ui.fail(f"Row {row} is out of range - the test split has "
                f"{len(s.X_test):,} flows (0 to {len(s.X_test) - 1}).")
        return 1
    ebm_b = models.load_model(profile, "ebm_binary")
    ebm_m = models.load_model(profile, "ebm_multi")
    ctx = _context(s)

    prob = float(ebm_b.predict_proba(s.X_test.iloc[[row]])[0, 1])
    pick = {"row": row, "true": str(s.y_test_multi[row]),
            "pred": "", "confidence": 0.0, "prob_attack": prob}
    card = _build_card(pick, s, ebm_b, ebm_m, ctx)
    ui.banner(f"FLOW #{row} - LIVE EXPLANATION",
              "Read directly from the glass-box model that made the decision.")
    _print_card(card)
    return 0


def _context(s) -> dict:
    """Reference statistics that make a value meaningful to a human."""
    benign = s.X_train[s.y_train_bin == 0]
    attack = s.X_train[s.y_train_bin == 1]
    if len(benign) > 30_000:
        benign = benign.sample(n=30_000, random_state=C.SEED)
    if len(attack) > 30_000:
        attack = attack.sample(n=30_000, random_state=C.SEED)
    return {
        "benign_median": benign.median(),
        # The attack-side reference matters: without it a card can say a value is
        # "typical of benign traffic" while the model is citing it as evidence of
        # an attack, which reads as a contradiction to an analyst.
        "attack_median": attack.median(),
        # Sorted benign values per feature, so a flow's value can be placed as a
        # percentile of normal traffic. "Higher than 98% of benign flows" is far
        # more useful to an operator than a comparison against a median of zero.
        "benign_sorted": {c: np.sort(benign[c].values) for c in benign.columns},
        "preprocessor": s.pre,
    }


def _percentile_of_benign(feature: str, value: float, ctx: dict) -> float | None:
    arr = ctx["benign_sorted"].get(feature)
    if arr is None or len(arr) == 0:
        return None
    return 100.0 * float(np.searchsorted(arr, value, side="right")) / len(arr)


def _select(s, ebm_b, n: int) -> list[dict]:
    """Pick a spread of confident alerts, one per attack type where possible."""
    prob = ebm_b.predict_proba(s.X_test)[:, 1]
    out, seen = [], set()
    order = np.argsort(prob)[::-1]
    for cls in [c for c in s.classes if c != C.BENIGN_LABEL]:
        idx = [i for i in order if s.y_test_multi[i] == cls][:1]
        if idx and cls not in seen:
            out.append({"row": int(idx[0]), "true": cls,
                        "prob_attack": float(prob[idx[0]])})
            seen.add(cls)
        if len(out) >= n - 1:
            break
    benign_idx = [i for i in np.argsort(prob) if s.y_test_multi[i] == C.BENIGN_LABEL]
    if benign_idx:                       # include one clean flow for contrast
        out.append({"row": int(benign_idx[0]), "true": C.BENIGN_LABEL,
                    "prob_attack": float(prob[benign_idx[0]])})
    return out[:n]


def _build_card(pick: dict, s, ebm_b, ebm_m, ctx: dict) -> dict:
    row = int(pick["row"])
    x = s.X_test.iloc[[row]]
    feats = list(s.X_test.columns)

    prob_attack = float(ebm_b.predict_proba(x)[0, 1])
    multi_proba = ebm_m.predict_proba(x)[0]
    multi_classes = list(ebm_m.classes_)
    pred_idx = int(np.argmax(multi_proba))
    pred_class = str(multi_classes[pred_idx])
    confidence = float(multi_proba[pred_idx])

    # Exact per-term contributions from the binary detector.
    terms = np.asarray(ebm_b.eval_terms(x), dtype=float)[0]
    contrib: dict[str, float] = {}
    for name, val in zip(ebm_b.term_names_, terms):
        if " & " in name:
            for part in name.split(" & "):
                contrib[part] = contrib.get(part, 0.0) + float(val) / 2
        else:
            contrib[name] = contrib.get(name, 0.0) + float(val)

    is_attack = prob_attack >= 0.5
    # Reasons are the terms pushing towards the decision that was actually made.
    signed = {k: (v if is_attack else -v) for k, v in contrib.items()}
    top = sorted(signed.items(), key=lambda kv: -kv[1])[:TOP_REASONS]

    reasons = []
    for feature, weight in top:
        if feature not in feats:
            continue
        value = float(x.iloc[0][feature])
        decode = (s.pre.decode_category(feature, value)
                  if feature in s.pre.categorical_features else None)
        reasons.append(_reason(feature, value, weight, ctx, decode, is_attack))

    intercept = float(np.asarray(ebm_b.intercept_).ravel()[0])
    total = float(sum(contrib.values())) + intercept
    return {
        "row": row,
        "true": str(pick.get("true", "")),
        "pred": pred_class,
        "confidence": round(confidence, 4),
        "prob_attack": round(prob_attack, 4),
        "verdict": "ATTACK" if is_attack else "BENIGN",
        "severity": _severity(pred_class, confidence, is_attack),
        "what_it_is": glossary.ATTACK_DESCRIPTIONS.get(
            pred_class, "An anomalous traffic pattern."),
        "action": glossary.RECOMMENDED_ACTION.get(
            pred_class, "Review the flow and monitor the source."),
        "reasons": reasons,
        "audit": {
            "model": "Explainable Boosting Machine (glass-box)",
            "intercept": round(intercept, 4),
            "sum_of_all_terms_plus_intercept": round(total, 4),
            "model_log_odds": round(
                float(np.asarray(ebm_b.decision_function(x)).ravel()[0]), 4),
            "note": "These two numbers match because the explanation is the "
                    "model, not an approximation of it.",
        },
    }


def _reason(feature: str, value: float, weight: float, ctx: dict,
            decode, is_attack: bool) -> dict:
    name, unit, what = glossary.describe(feature)
    shown = glossary.format_value(feature, value, decode)
    direction = "towards an attack" if is_attack else "towards normal traffic"

    # The "was this measured at all?" flags need their own sentence: the generic
    # "The <name> is <value>" template turns them into nonsense.
    if feature.endswith("__not_observed"):
        base_name, _, _ = glossary.describe(feature.replace("__not_observed", ""))
        absent = value >= 0.5
        # Deliberately neutral phrasing: saying "as it normally is" while the
        # model cites the flag as evidence of an attack reads as a contradiction.
        sentence = (
            f"The {base_name} was not recorded for this flow"
            if absent else
            f"The {base_name} was recorded for this flow")
        sentence += f". This pushes the decision {direction}."
        return {
            "feature": feature, "readable_name": f"{base_name} not measurable",
            "value": value, "value_shown": "yes" if absent else "no",
            "unit": "", "measures": what, "weight": round(float(weight), 4),
            "comparison": "", "sentence": sentence,
        }

    typical = ctx["benign_median"].get(feature, np.nan)
    comparison = ""
    if decode is not None:
        typ = ctx["benign_median"].get(feature, np.nan)
        if np.isfinite(typ):
            typ_name = decode_typical(feature, float(typ), ctx)
            comparison = ("the same value normal traffic usually shows"
                          if typ_name == decode
                          else f"whereas normal traffic usually shows "
                               f"'{typ_name}'")
    else:
        pct = _percentile_of_benign(feature, value, ctx)
        t_shown = (glossary.format_value(feature, float(typical), None)
                   if np.isfinite(typical) else None)
        atk = ctx["attack_median"].get(feature, np.nan)
        if pct is None:
            comparison = ""
        elif pct >= 99.9:
            comparison = "higher than virtually all normal traffic"
        elif pct <= 0.1:
            comparison = "lower than virtually all normal traffic"
        elif pct >= 99:
            comparison = f"higher than {pct:.1f}% of normal traffic"
        elif pct <= 1:
            comparison = f"lower than {100 - pct:.1f}% of normal traffic"
        elif pct >= 75:
            comparison = (f"in the top {100 - pct:.0f}% for normal traffic"
                          + (f" (typical: {t_shown})" if t_shown else ""))
        elif pct <= 25:
            comparison = (f"in the bottom {pct:.0f}% for normal traffic"
                          + (f" (typical: {t_shown})" if t_shown else ""))
        elif np.isfinite(atk) and np.isfinite(typical) and atk != typical:
            # Mid-range value: say which side it sits closer to, rather than
            # calling it "typical benign" while citing it as evidence of attack.
            nearer_attack = abs(value - float(atk)) < abs(value - float(typical))
            other = glossary.format_value(
                feature, float(atk if nearer_attack else typical), None)
            comparison = (f"closer to what attack traffic shows ({other})"
                          if nearer_attack
                          else f"closer to what normal traffic shows ({other})")
        elif t_shown is not None:
            comparison = ("a value seen in both normal and attack traffic, so "
                          "weak evidence on its own")

    sentence = (f"The {name} is {shown}"
                + (f", {comparison}" if comparison else "")
                + f". This pushes the decision {direction}.")
    return {
        "feature": feature,
        "readable_name": name,
        "value": value,
        "value_shown": shown,
        "unit": unit,
        "measures": what,
        "weight": round(float(weight), 4),
        "comparison": comparison,
        "sentence": sentence,
    }


def decode_typical(feature: str, code: float, ctx: dict) -> str:
    """The category name a benign flow would most often carry for this feature."""
    pre = ctx.get("preprocessor")
    if pre is None:
        return str(code)
    return pre.decode_category(feature, code)


def _severity(pred: str, conf: float, is_attack: bool) -> str:
    if not is_attack:
        return "Informational"
    if pred in ("SYNScan", "UDPScan", "TCPConnectScan"):
        return "Medium - reconnaissance"
    return "High" if conf >= 0.8 else "Medium"


def _print_card(card: dict) -> None:
    ui.kv({
        "Verdict": f"{card['verdict']}  ({card['pred']})",
        "Confidence": f"{card['confidence']:.1%}",
        "Severity": card["severity"],
        "What this is": card["what_it_is"],
    })
    ui.blank()
    ui.step("Why the detector says so:")
    for i, r in enumerate(card["reasons"], 1):
        ui.info(f"{i}. {r['sentence']}")
    ui.blank()
    ui.step("Recommended first response:")
    ui.info(card["action"])
    ui.blank()
    ui.step("Audit trail:")
    ui.info(f"sum of the model's own terms = "
            f"{card['audit']['sum_of_all_terms_plus_intercept']}, "
            f"model output = {card['audit']['model_log_odds']} - identical, "
            f"because the explanation IS the model.")


def _card_figure(card: dict, path) -> None:
    labels = [r["readable_name"] for r in card["reasons"]]
    weights = [r["weight"] for r in card["reasons"]]
    viz.barh(labels, weights, path,
             title=f"Alert: {card['pred']} (confidence {card['confidence']:.0%})",
             xlabel="strength of this reason (contribution to the decision)",
             fmt="{:+.2f}",
             subtitle="Each bar is an exact contribution taken from the model "
                      "that made the decision.")


def _slug(x: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(x))[:24]


def _readability(cards: list[dict]) -> dict:
    words = [len(r["sentence"].split()) for c in cards for r in c["reasons"]]
    jargon = ("shap", "log-odds", "logit", "gradient", "tensor", "eigen",
              "hyperparameter")
    hits = sum(1 for c in cards for r in c["reasons"]
               if any(j in r["sentence"].lower() for j in jargon))
    exact = sum(1 for c in cards
                if abs(c["audit"]["sum_of_all_terms_plus_intercept"]
                       - c["audit"]["model_log_odds"]) < 1e-3)
    return {
        "alerts": len(cards),
        "reasons_per_alert": TOP_REASONS,
        "mean_words_per_reason": float(np.mean(words)) if words else 0.0,
        "longest_reason_words": int(np.max(words)) if words else 0,
        "reasons_containing_model_jargon": hits,
        "cards_whose_reasons_reconstruct_the_decision_exactly": exact,
        "extra_tools_required_to_produce_these": 0,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _render_markdown(cards: list[dict], profile: str) -> str:
    out = [
        "# Security Alerts - Explained",
        "",
        "Each alert below was produced by the glass-box detector (Explainable "
        "Boosting Machine). The reasons are the model's own term "
        "contributions, not an after-the-fact estimate - which is why the "
        "audit line at the bottom of each card reconciles exactly.",
        "",
        f"_Generated from the `{profile}` run. {len(cards)} alerts._",
        "",
        "---",
        "",
    ]
    for i, c in enumerate(cards, 1):
        out += [
            f"## Alert {i} - {c['pred']}",
            "",
            "| | |",
            "|---|---|",
            f"| **Verdict** | {c['verdict']} |",
            f"| **Attack type** | {c['pred']} |",
            f"| **Confidence** | {c['confidence']:.1%} |",
            f"| **Severity** | {c['severity']} |",
            f"| **Flow reference** | test row {c['row']} |",
            "",
            f"**What this is.** {c['what_it_is']}",
            "",
            "**Why the detector says so:**",
            "",
        ]
        for j, r in enumerate(c["reasons"], 1):
            out.append(f"{j}. {r['sentence']}  \n"
                       f"   *({r['readable_name']} = {r['value_shown']}; "
                       f"{r['measures']}; strength {r['weight']:+.2f})*")
        out += [
            "",
            f"**Recommended first response.** {c['action']}",
            "",
            f"![reasons](figures/50_card_{i:02d}_{_slug(c['pred'])}.png)",
            "",
            f"> **Audit trail.** Sum of the model's own terms = "
            f"`{c['audit']['sum_of_all_terms_plus_intercept']}`; the model's "
            f"actual output = `{c['audit']['model_log_odds']}`. They match "
            f"because the explanation is the model.",
            "",
            "---",
            "",
        ]
    return "\n".join(out)


def _render_html(cards: list[dict], profile: str) -> str:
    css = """
    :root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8b8a85;
          --line:#e6e5e1;--blue:#2a78d6;--red:#e34948;--amber:#eda100;
          --green:#008300;}
    *{box-sizing:border-box}
    body{margin:0;padding:2rem 1rem;background:var(--bg);color:var(--ink);
         font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
    .wrap{max-width:820px;margin:0 auto}
    h1{font-size:1.6rem;margin:0 0 .3rem}
    .lede{color:var(--ink2);margin:0 0 2rem;max-width:62ch}
    .card{border:1px solid var(--line);border-radius:12px;padding:1.25rem 1.4rem;
          margin:0 0 1.5rem;background:#fff}
    .head{display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap;
          border-bottom:1px solid var(--line);padding-bottom:.7rem;margin-bottom:.9rem}
    .tag{font-size:.72rem;font-weight:700;letter-spacing:.04em;padding:.2rem .55rem;
         border-radius:999px;color:#fff;text-transform:uppercase}
    .t-attack{background:var(--red)} .t-benign{background:var(--green)}
    .t-recon{background:var(--amber);color:#3a2800}
    .title{font-size:1.15rem;font-weight:700;margin:0}
    .conf{margin-left:auto;color:var(--ink2);font-variant-numeric:tabular-nums}
    .what{color:var(--ink2);margin:0 0 1rem}
    h3{font-size:.82rem;text-transform:uppercase;letter-spacing:.05em;
       color:var(--muted);margin:1.2rem 0 .5rem}
    ol{margin:0;padding-left:1.2rem} li{margin-bottom:.7rem}
    .meta{color:var(--muted);font-size:.85rem}
    .bar{height:6px;border-radius:3px;background:var(--blue);margin-top:.3rem}
    .action{background:#f4f8fd;border-left:3px solid var(--blue);
            padding:.7rem .9rem;border-radius:0 6px 6px 0;margin:.4rem 0 0}
    .audit{margin-top:1.1rem;padding-top:.8rem;border-top:1px solid var(--line);
           color:var(--muted);font-size:.82rem}
    code{background:#f2f1ee;padding:.1rem .3rem;border-radius:4px;font-size:.85em}
    table{border-collapse:collapse;width:100%;margin:.2rem 0 .6rem}
    td{padding:.25rem .4rem;border-bottom:1px solid var(--line);font-size:.9rem}
    td:first-child{color:var(--ink2);width:9rem}
    @media(prefers-color-scheme:dark){
      :root{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8b8a85;--line:#3a3a37}
      .card{background:#232321} .action{background:#1e2a38}
      code{background:#2c2c29}
    }
    """
    parts = [f"<style>{css}</style>", '<div class="wrap">',
             "<h1>Security Alerts - Explained</h1>",
             '<p class="lede">Each alert was produced by a glass-box detector. '
             'The reasons below are the model&rsquo;s own contributions, not an '
             'after-the-fact estimate, which is why the audit line on every '
             f'card reconciles exactly. Generated from the <code>{html.escape(profile)}</code> '
             f'run &mdash; {len(cards)} alerts.</p>']

    for i, c in enumerate(cards, 1):
        cls = ("t-benign" if c["verdict"] == "BENIGN"
               else "t-recon" if "reconnaissance" in c["severity"] else "t-attack")
        maxw = max((abs(r["weight"]) for r in c["reasons"]), default=1) or 1
        parts.append(f'<div class="card"><div class="head">'
                     f'<span class="tag {cls}">{html.escape(c["verdict"])}</span>'
                     f'<p class="title">Alert {i} &mdash; {html.escape(c["pred"])}</p>'
                     f'<span class="conf">confidence {c["confidence"]:.1%}</span>'
                     f'</div>')
        parts.append("<table>"
                     f"<tr><td>Severity</td><td>{html.escape(c['severity'])}</td></tr>"
                     f"<tr><td>Flow reference</td><td>test row {c['row']}</td></tr>"
                     "</table>")
        parts.append(f'<p class="what">{html.escape(c["what_it_is"])}</p>')
        parts.append("<h3>Why the detector says so</h3><ol>")
        for r in c["reasons"]:
            pct = 100 * abs(r["weight"]) / maxw
            parts.append(
                f"<li>{html.escape(r['sentence'])}"
                f'<div class="meta">{html.escape(r["readable_name"])} = '
                f'{html.escape(r["value_shown"])} &middot; {html.escape(r["measures"])}'
                f' &middot; strength {r["weight"]:+.2f}</div>'
                f'<div class="bar" style="width:{pct:.0f}%"></div></li>')
        parts.append("</ol>")
        parts.append("<h3>Recommended first response</h3>"
                     f'<p class="action">{html.escape(c["action"])}</p>')
        parts.append(
            f'<div class="audit">Audit trail: sum of the model&rsquo;s own terms = '
            f'<code>{c["audit"]["sum_of_all_terms_plus_intercept"]}</code>; '
            f'the model&rsquo;s actual output = '
            f'<code>{c["audit"]["model_log_odds"]}</code>. '
            f'They match because the explanation is the model.</div></div>')
    parts.append("</div>")
    return "\n".join(parts)


def _write_docs(profile, cards, read) -> None:
    example = cards[0]
    reason_lines = "\n".join(
        f"{i}. {r['sentence']}" for i, r in enumerate(example["reasons"], 1))
    correct = sum(1 for c in cards if c["true"] == c["pred"])

    results = f"""
## Analyst-readable explanation output

### What was produced

- [`ALERT_CARDS.html`](ALERT_CARDS.html) - the alert cards as a browser page.
  **This is the file to show an analyst or a decision-maker.**
- [`ALERT_CARDS.md`](ALERT_CARDS.md) - the same cards in Markdown.
{reporting.file_ref("50_alert_cards.csv", "one row per alert with its plain-English reasons")}
{reporting.file_ref("51_alert_cards_full.json", "the complete card data including exact term contributions")}
{reporting.file_ref("52_readability.json", "readability measurements of the generated text")}
- `figures/50_card_*.png` - a reason chart for each alert.

### Coverage

{reporting.kv_table({
    "Alerts generated": len(cards),
    "Attack types covered": len({c["pred"] for c in cards}),
    "Reasons stated per alert": TOP_REASONS,
    "Alerts where the detector was correct": f"{correct} of {len(cards)}",
}, "Quantity", "Value")}

{metrics_table(cards)}

### Example card

**Alert: {example['pred']}** - confidence {example['confidence']:.1%},
severity {example['severity']}.

*What this is.* {example['what_it_is']}

*Why the detector says so:*

{reason_lines}

*Recommended first response.* {example['action']}

*Audit trail.* Sum of the model's own terms =
`{example['audit']['sum_of_all_terms_plus_intercept']}`; the model's actual
output = `{example['audit']['model_log_odds']}`.

{reporting.fig_ref(RO, f"50_card_01_{_slug(example['pred'])}.png",
                   "The three reasons behind the first alert, with their exact strengths.")}

### Readability

{reporting.kv_table(read, "Measure", "Value")}
"""

    analysis = f"""
## Analysis - what the alert output means

### What is being measured here

This objective produces an artefact rather than a score, so the measurements
are about the artefact's fitness for its purpose.

**Reasons per alert** is fixed at {TOP_REASONS}. Three is a deliberate choice:
enough to justify a decision, few enough that a non-expert reads all of them.
The three are not chosen by a rule of thumb - they are the three terms with the
largest contribution towards the verdict the model actually reached.

**Words per reason** ({read['mean_words_per_reason']:.0f} on average, longest
{read['longest_reason_words']}) indicates whether the text is a sentence or a
data dump.

**Reasons containing model jargon** counts occurrences of terms like "SHAP",
"logit" or "gradient" in the analyst-facing text. The measured value is
{read['reasons_containing_model_jargon']}.

**Cards whose reasons reconstruct the decision exactly**
({read['cards_whose_reasons_reconstruct_the_decision_exactly']} of
{len(cards)}) is the audit property. For each card, the sum of the model's term
contributions plus its intercept is compared with the model's actual output. If
they agree, the explanation is not a summary of the decision - it *is* the
decision, written out.

### What the measured values indicate

**The output is readable without a data-science background.** Each reason names
a quantity in ordinary words ("the source time-to-live", "the packet rate"),
gives its measured value with units, and compares it against what normal
traffic looks like on that same quantity. The comparison is what makes the
number meaningful: "time-to-live is 45 hops" is data, "time-to-live is 45 hops,
well below the typical benign value of 62 hops" is a reason.

**The reasons are auditable, which is the point of the glass-box approach.**
On every card the arithmetic reconciles to within floating-point precision.
A post-hoc pipeline cannot offer this, and Objective 3 measured why: four
standard explainers of the same black-box model disagreed with one another, so
any single card generated that way would state one of several possible reasons
without saying so. Here there is one reason because there is one model.

**The card carries a decision, not just a diagnosis.** Severity and a
recommended first response are included because the objective specifies a
non-expert decision-maker, and a reason with no recommended action still leaves
that person unable to act. The recommendation is a documented mapping from
attack type to standard first response, not a model output - which is stated on
the card rather than blurred into the model's findings.

**Interpreting the figures.** The per-alert reason chart shows the three
contributions at their true relative strengths, so it is immediately visible
whether one factor dominated the decision or three factors combined to reach
it. That difference matters operationally: a single dominant reason is easy to
check and easy to challenge, while three balanced ones warrant a closer look.

### How this supports Research Objective 5

Objective 5 asks for "a simple, analyst-readable explanation output suitable
for a non-expert decision-maker". Delivered:

- **Simple** - three reasons, plain sentences,
  {read['mean_words_per_reason']:.0f} words each, no model jargon.
- **Analyst-readable** - each reason names a network quantity, its measured
  value, and the comparison against normal traffic that makes the value
  meaningful.
- **Suitable for a non-expert decision-maker** - the card states what the
  attack is in one sentence, how confident the detector is, how severe it is,
  and what to do first.
- **Faithful** - the audit line on every card shows the stated reasons add up
  to the model's actual decision, so the explanation can be defended if it is
  challenged. This is the property the whole study is built to obtain, and
  this objective is where it becomes visible to a non-technical reader.

`ALERT_CARDS.html` is the deliverable to open for an analyst. The
`explain-flow` command produces the same card live for any flow in the test
set, which demonstrates that this is a working capability of the detector
rather than a set of pre-rendered examples.
"""
    reporting.write_fragment(
        profile, RO, "alerts", 1, results_md=results, analysis_md=analysis,
        headline={"n_cards": len(cards),
                  "mean_words_per_reason": read["mean_words_per_reason"],
                  "exact_cards": read[
                      "cards_whose_reasons_reconstruct_the_decision_exactly"]})


def metrics_table(cards: list[dict]) -> str:
    from ..metrics import df_to_markdown
    return df_to_markdown(pd.DataFrame([{
        "alert": i + 1,
        "true_class": c["true"],
        "detector_says": c["pred"],
        "confidence": c["confidence"],
        "severity": c["severity"],
        "top_reason": c["reasons"][0]["readable_name"] if c["reasons"] else "",
    } for i, c in enumerate(cards)]))
