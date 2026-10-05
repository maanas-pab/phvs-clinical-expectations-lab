"""Build the static GitHub Pages site from the same modules the app uses.

Every figure and number here is produced by ``phvs_lab.modules`` - the same
functions Streamlit calls - so the published page cannot drift from the app.

The page follows the thesis order: recommendation, three research questions,
the three slide-ready exhibits, the analysis underneath them, valuation, and
then limitations.

Usage: ``.venv/bin/python build_site.py`` writes ``docs/index.html``.
"""

from __future__ import annotations

import datetime as dt
import html as html_lib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from phvs_lab import config as cf
from phvs_lab.modules import data_loader as dl
from phvs_lab.modules import economics as ec
from phvs_lab.modules import evidence as ev
from phvs_lab.modules import prescribing_map as pm
from phvs_lab.modules import research as rs
from phvs_lab.modules import response_engine as re_
from phvs_lab.modules import trial_audit as ta
from phvs_lab.modules import valuation as val
from phvs_lab.utils import visualization as viz

REPO = Path(__file__).resolve().parent
OUT = REPO / "docs" / "index.html"
EXHIBIT_DIR = REPO / "docs" / "exhibits"

PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# --------------------------------------------------------------------------
# Markdown -> HTML (headings, tables, lists, code, emphasis)
# --------------------------------------------------------------------------

def _inline(text: str) -> str:
    text = html_lib.escape(text, quote=False)
    text = (text.replace("&amp;#8805;", "\u2265")
                .replace("&amp;nbsp;", " "))
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def md_to_html(text: str) -> str:
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            out.append("<pre><code>" + html_lib.escape("\n".join(block)) + "</code></pre>")
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            head = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            th = "".join(f"<th>{_inline(c)}</th>" for c in head)
            body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                           for r in rows)
            out.append(f"<table class='tbl'><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>")
            continue
        m = re.match(r"^(#{2,4})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f"<h{lvl} id='{slug}'>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if line.startswith("> "):
            out.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
            i += 1
            continue
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            out.append("<ol>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + "</ol>")
            continue
        if not line.strip():
            i += 1
            continue
        para = []
        start = i
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", ">", "```")) \
                and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
            para.append(lines[i])
            i += 1
        if i == start:
            out.append(f"<p>{_inline(lines[i])}</p>")
            i += 1
        else:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chart(fig, caption: str = "") -> str:
    fig.update_layout(autosize=True, width=None,
                      **{k: v for k, v in cf.PLOT_LAYOUT.items() if k in ("font", "legend")},
                      paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    body = fig.to_html(include_plotlyjs=False, full_html=False,
                       config={"displayModeBar": False, "responsive": True})
    cap = f"<figcaption>{html_lib.escape(caption)}</figcaption>" if caption else ""
    return f"<figure>{body}{cap}</figure>"


def table(df: pd.DataFrame, max_rows: int = 0) -> str:
    view = df.head(max_rows) if max_rows else df
    rendered = view.to_html(index=False, border=0, escape=True, classes="tbl")
    note = "" if not max_rows or len(view) == len(df) else \
        f"<p class='note'>Showing {len(view)} of {len(df)} rows.</p>"
    return f"<div class='tablewrap'>{rendered}</div>{note}"


def metric(label: str, value: str, sub: str = "") -> str:
    s = f"<div class='msub'>{html_lib.escape(sub)}</div>" if sub else ""
    return (f"<div class='metric'><div class='mlabel'>{html_lib.escape(label)}</div>"
            f"<div class='mvalue'>{value}</div>{s}</div>")


def panel(kicker: str, title: str, body: str, anchor: str | None = None) -> str:
    id_attr = f' id="{anchor}"' if anchor else ""
    return (f'<section class="panel"{id_attr}><div class="kicker">{html_lib.escape(kicker)}</div>'
            f'<h2>{html_lib.escape(title)}</h2>{body}</section>')


def _pct_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = [f"{100 * float(v):.1f}%" for v in out[c]]
    return out


def _unit_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    units = out["unit"].astype(str)
    for c in cols:
        out[c] = out[c].astype(object)
    locs = {c: out.columns.get_loc(c) for c in cols}
    for i, unit in enumerate(units):
        for c in cols:
            v = float(df.iloc[i][c])
            if "USD" in unit:
                s = f"${v:,.0f}"
            elif "multiple" in unit:
                s = f"{v:.2f}\u00d7"
            else:
                s = f"{100 * v:.1f}%"
            out.iat[i, locs[c]] = s
    return out


def fmt_rq_table(df: pd.DataFrame) -> pd.DataFrame:
    if "unit" in df.columns:
        cols = [c for c in ("stated", "required_for_price", "value") if c in df.columns]
        return _unit_cols(df, cols)
    if "value" in df.columns:
        return _pct_cols(df, ["value"])
    return df


def fmt_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    if "unit" in df.columns:
        return _unit_cols(df, ["value"])
    return _pct_cols(df, ["value"])


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def research_block_html(block: "rs.ResearchBlock") -> str:
    metrics = "".join(metric(k, v) for k, v in block.metrics.items())
    steps = ""
    for label, text in (("Methodology", block.methodology),
                        ("Finding", block.finding),
                        ("Investment implication", block.implication),
                        ("Valuation impact", block.valuation_impact)):
        steps += (f'<h4>{label}</h4><div class="step">{md_to_html(text)}</div>')
    tables = ""
    if block.table is not None:
        tables += f"<h4>Supporting table</h4>{table(fmt_rq_table(block.table))}"
    if block.thresholds is not None:
        tables += f"<h4>Thresholds and break-evens</h4>{table(fmt_thresholds(block.thresholds))}"
    return (
        f'<div class="rq" id="{block.key.lower()}">'
        f'<div class="kicker">{block.key} \u00b7 status: {block.status}</div>'
        f'<h3>{_inline(block.question)}</h3>'
        f'<div class="metrics">{metrics}</div>{steps}{tables}</div>'
    )


def exhibit_html(ex: Dict) -> str:
    png = EXHIBIT_DIR / f"{ex['png']}.png"
    if png.exists():
        img = (f'<img class="exhibit" src="exhibits/{ex["png"]}.png" '
               f'alt="{html_lib.escape(ex["title"])}" loading="lazy">')
    else:
        img = f"<p class='note'>Exhibit image not found at <code>docs/{png.name}</code>.</p>"
    sources = "".join(f"<li>{_inline(line)}</li>" for line in rs.exhibit_source_lines(ex))
    return (
        f'<div class="rq">'
        f'<div class="kicker">{ex["id"]} \u00b7 {ex["rq"]} \u00b7 {html_lib.escape(ex["status"])}</div>'
        f'<h3>{html_lib.escape(ex["title"])}</h3>'
        f'<p class="headline">{html_lib.escape(ex["headline"])}</p>'
        f'{img}'
        f'<p class="note"><strong>Finding.</strong> {_inline(ex["caption"])}</p>'
        f'<h4>Methodology</h4><div class="step">{md_to_html(ex["methodology"])}</div>'
        f'<h4>What it means for the thesis</h4><div class="step">{md_to_html(ex["implication"])}</div>'
        f'<h4>Sources</h4><ul class="src">{sources}</ul>'
        f'</div>'
    )


def figure_list() -> list[tuple[str, str, str, list[str]]]:
    """(group, title, chart html, notes) - the analysis underneath the exhibits."""
    base = val.params_from_assumptions()
    price = base.stock_price
    trials = dl.get_clinical_trials()
    repro = val.reproduce_pitch_valuation(dl.get_valuation_model())
    summary = val.value_summary(base)
    sens = val.sensitivity_ranking(base)
    impl = val.market_implied(base)
    dist = list(re_.distributions_for_trial("CHAPTER-3", trials).values())[0]
    window_curves = ta.window_sensitivity_curves([0.4, 0.8], [0.3, 0.5, 1.0],
                                                 [30, 90, 168, 182, 365])
    adjusted = ta.window_adjusted_table(trials, target_days=180, dispersion=0.5)
    surface = pm.value_surface(np.linspace(0.6, 1.5, 19), np.linspace(0.02, 0.50, 19))
    ledger = ec.population_ledger()
    params = ec.summarize_economics()["params"]
    targets = impl.set_index("input")

    def pctf(x, d=1):
        return f"{x * 100:.{d}f}%"

    return [
        ("RQ1 \u00b7 response", "Chapter 3 response reconstruction",
         chart(viz.fig_response_categories(dist), dist.window_note()),
         ["Exclusive categories are subtraction of the published cumulative thresholds; they are "
          "only point identified when all four thresholds are disclosed."]),
        ("RQ1 \u00b7 response", "Exact intervals on the disclosed thresholds",
         chart(viz.fig_response_thresholds(dist.proportion_intervals()),
               "Clopper-Pearson intervals: sampling uncertainty on the observed counts, stated "
               "separately from window length."),
         []),
        ("RQ1 \u00b7 response", "Window mechanics",
         chart(viz.fig_window_sensitivity(window_curves),
               "Attack-free is a count over a stated window and falls mechanically as the "
               "window lengthens at an unchanged rate."),
         []),
        ("RQ1 \u00b7 response", "Window-standardised comparison",
         chart(viz.fig_window_adjustment(adjusted),
               "A gamma-Poisson standardisation to a common window. This is a stated "
               "transformation of published counts, not a re-analysis and not a randomised "
               "comparison between separate trials."),
         []),
        ("RQ2 \u00b7 adoption", "Population funnel",
         chart(viz.fig_population_funnel(ledger),
               "Segments are mutually exclusive and validated to sum to 1.0 of the eligible "
               "pool."),
         []),
        ("RQ2 \u00b7 adoption", "Prevention and rescue",
         chart(viz.fig_prevention_ladder(ec.prevention_rescue_ladder(params)),
               "Attacks, patients and revenue are three separate quantities with their own "
               "conversion assumptions."),
         []),
        ("RQ2 \u00b7 adoption", "Efficacy to breakthrough cases",
         chart(viz.fig_efficacy_projection(
             ec.efficacy_projection(params, np.linspace(0.40, 1.00, 25)),
         ), f"Dispersion k={ec.bridge_components(params)['fitted_dispersion_k']:.4g} calibrated "
            "once from the trial's published values, then held fixed while efficacy varies."),
         []),
        ("RQ2 \u00b7 adoption", "Prescribing map to value per share",
         chart(viz.fig_prescribing_contour(surface, price,
                                           pm.MapParams().switching_rate, 1.0),
               "Segments feed a logit choice model; switchers accumulate at the stated annual "
               "switching rate over the time to peak. The contour reads in value per share."),
         [f"Stated inputs land at "
          f"{pm.value_for_inputs(1.0, pm.MapParams().switching_rate)['implied_peak_penetration'] * 100:.1f}% "
          f"peak penetration and "
          f"${pm.value_for_inputs(1.0, pm.MapParams().switching_rate)['value_per_share']:,.2f} per share. "
          f"Break-even switching is "
          f"{pm.break_even_switching(1.0)['required_switching_rate'] * 100:.1f}% a year."]),
        ("RQ2 \u00b7 adoption", "Choice shares by segment",
         chart(viz.fig_choice_shares(pm.choice_probabilities()),
               "Stay on current prophylaxis, switch to an existing oral, switch to PHVS oral. "
               "Coefficients are ledger assumptions, not measured preferences."),
         []),
        ("RQ3 \u00b7 valuation", "The pitch, reproduced first",
         chart(viz.fig_pitch_vs_models(repro, val.scenario_table(base), price),
               "The pitch's own cash flows re-run on its terms, then on the verified 70.2m "
               "share count and verified net cash, against the observed price."),
         ["Share count, terminal omission and embedded-price findings are listed, not silently "
          "corrected."]),
        ("RQ3 \u00b7 valuation", "Equity bridge",
         chart(viz.fig_equity_bridge(summary),
               "Enterprise value plus verified net cash of "
               f"${base.net_cash:,.0f} equals equity value; per share on 70.2m shares."),
         []),
        ("RQ3 \u00b7 valuation", "Present value by year",
         chart(viz.fig_pv_profile(val.build_cashflows(base)),
               "Fixed costs at face value; product contributions weighted once by approval "
               "probability; terminal growth forced to zero after the loss-of-exclusivity year."),
         []),
        ("RQ3 \u00b7 valuation", "What the price requires, one input at a time",
         chart(viz.fig_required_inputs(impl, sens),
               "Each input solved separately so the model equals the observed price. The price "
               "is a bundle, so these cannot all be true at once."),
         [f"Peak penetration {pctf(targets.loc['peak_penetration_prophylaxis', 'required_value'])} "
          f"against {pctf(base.peak_penetration_prophylaxis)} stated; approval "
          f"{pctf(targets.loc['approval_prophylaxis', 'required_value'])} against "
          f"{pctf(base.approval_prophylaxis)}; discount "
          f"{pctf(targets.loc['discount_rate', 'required_value'])} against "
          f"{pctf(base.discount_rate)}."]),
        ("RQ3 \u00b7 valuation", "One input shocked at a time",
         chart(viz.fig_tornado(sens, summary["value_per_share"]),
               "Sensitivity of value per share to a single shock, all else held at the "
               "reference case."),
         []),
        ("RQ3 \u00b7 valuation", "Scenarios and the short",
         chart(viz.fig_short_scenarios(val.short_table(base)),
               "Net of a stated annual borrow cost over the stated horizon. Not a "
               "recommendation."),
         []),
        ("Provenance", "Evidence coverage",
         chart(viz.fig_evidence_coverage(ev.provenance_summary(trials)),
               "Every observation is tagged with its source and verification status."),
         []),
    ]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build() -> Path:
    base = val.params_from_assumptions()
    price = base.stock_price
    repro = val.reproduce_pitch_valuation(dl.get_valuation_model())
    scenarios = val.scenario_table(base)
    summary = val.value_summary(base)
    bundle = val.required_bundle(base)
    rec = rs.short_recommendation()
    impl = val.market_implied(base)
    targets = impl.set_index("input")
    assumptions = dl.get_commercial_assumptions()

    def pctf(x, d=1):
        return f"{x * 100:.{d}f}%"

    def usdf(x):
        return f"${x:,.0f}"

    # --- hero + headline metrics -----------------------------------------
    metrics = "".join([
        metric("Recommendation", f"{rec['recommendation']} {rec['ticker']}",
               f"{usdf(price)} close, {rec['price_date']}"),
        metric("Our stated-input model", f"${rec['model_value']:,.2f}",
               f"{rec['model_vs_price']:+.1%} vs the observed price"),
        metric("Enterprise value the price requires", usdf(rec["required_ev"]),
               f"{rec['required_multiple_of_reference']:.2f}\u00d7 the reference model, "
               f"{rec['required_multiple_of_cashflow_model']:.2f}\u00d7 the cash-flow model"),
        metric("Invalidation level", f"${rec['breakeven_value_per_share']:,.2f}",
               f"model value at or above this stops the position out "
               f"({pctf(rec['carry_pct'], 0)} carry)"),
        metric("Required peak penetration",
               pctf(targets.loc["peak_penetration_prophylaxis", "required_value"]),
               f"stated {pctf(base.peak_penetration_prophylaxis)}"),
        metric("Required approval probability",
               pctf(targets.loc["approval_prophylaxis", "required_value"]),
               f"stated {pctf(base.approval_prophylaxis)}"),
        metric("Required discount rate",
               pctf(targets.loc["discount_rate", "required_value"]),
               f"stated {pctf(base.discount_rate)}"),
        metric("Bear / bull per share",
               f"${scenarios.set_index('scenario').loc['bear', 'value_per_share']:,.2f} / "
               f"${scenarios.set_index('scenario').loc['bull', 'value_per_share']:,.2f}",
               f"{rec['spread_multiple']:.1f}\u00d7 spread, all of it unmeasured inputs"),
    ])

    required = (
        f"{usdf(bundle['required_enterprise_value'])} of enterprise value - "
        f"{bundle['required_multiple_of_stated_ev']:.2f}\u00d7 the extended model and "
        f"{repro['multiple_required']:.2f}\u00d7 the reproduced cash-flow model - reached at "
        f"{pctf(targets.loc['peak_penetration_prophylaxis', 'required_value'])} peak "
        f"prophylaxis penetration, a "
        f"{pctf(targets.loc['approval_prophylaxis', 'required_value'])} approval probability "
        f"or a {pctf(targets.loc['discount_rate', 'required_value'])} discount rate, each "
        f"holding everything else at its stated value"
    )
    supported = (
        f"the ledger's stated inputs, worth ${summary['value_per_share']:,.2f} a share; the "
        f"pitch's own cash flows, which reproduce to only "
        f"${repro['value_per_share_current_shares']:,.2f} on the verified "
        f"{repro['shares_current'] / 1e6:.1f}m shares; and the choice model at stated "
        f"preferences, which delivers {pctf(pm.implied_peak_penetration())} peak penetration"
    )
    unmeasured = ("real-world switching rates, realised net price after gross-to-net, "
                  "durability beyond 24 weeks and any head-to-head evidence against injected "
                  "prophylaxis")
    conclusion = cf.CONCLUSION_TEMPLATE.format(
        REC=rec["recommendation"], PRICE=f"${price:,.2f}", X=required, Y=supported, Z=unmeasured,
        W=f"${rec['breakeven_value_per_share']:,.2f}").replace("<b>", "**").replace("</b>", "**")

    # --- findings ---------------------------------------------------------
    findings = "".join(
        f'<li><strong>{html_lib.escape(f["headline"])}</strong> - {_inline(f["claim"])}'
        f'<div class="src">basis: {_inline(f["basis"])} \u00b7 status: {f["status"]}</div></li>'
        for f in rec["findings"])

    # --- research questions ----------------------------------------------
    rqs = "".join(research_block_html(b)
                  for b in (rs.rq1_response(), rs.rq2_adoption(), rs.rq3_price()))
    workstreams = rs.workstreams()

    # --- exhibits ---------------------------------------------------------
    exhibits = rs.exhibits()
    missing = [e["png"] for e in exhibits
               if not (EXHIBIT_DIR / f"{e['png']}.png").exists()]
    exhibit_body = "".join(exhibit_html(e) for e in exhibits)

    # --- analysis figures, grouped ---------------------------------------
    groups: dict[str, list[tuple[str, str, list[str]]]] = {}
    for group, title, fig_html, notes in figure_list():
        groups.setdefault(group, []).append((title, fig_html, notes))

    fig_sections = []
    for gi, (group, items) in enumerate(groups.items(), start=1):
        body = ""
        for j, (title, fig_html, notes) in enumerate(items, start=1):
            notes_html = "".join(f"<p class='note'>{_inline(n)}</p>" for n in notes)
            body += (f"<div class='chart'><div class='kicker'>Chart {gi}.{j}</div>"
                     f"<h3>{html_lib.escape(title)}</h3>{notes_html}{fig_html}</div>")
        fig_sections.append(panel(group, group.split(" \u00b7 ")[-1].capitalize(), body,
                                  anchor=slugify(group)))

    # --- limitations ------------------------------------------------------
    weakest = "".join(f"<h4>{html_lib.escape(t)}</h4><div class='step'>{md_to_html(txt)}</div>"
                      for t, txt in rs.limitations_notes())
    disconfirming = table(rs.disconfirming_evidence())
    invalid = val.invalidation_conditions(base)

    # --- audit ------------------------------------------------------------
    audit = ev.audit_report()
    if audit["findings"]:
        findings_html = "<ul>" + "".join(
            f"<li>{_inline(str(f))}</li>" for f in audit["findings"]) + "</ul>"
    else:
        findings_html = ("<p class='note'>No failed verification checks: every stored count "
                         "that could be re-checked against the registry matched.</p>")
    audit_html = (
        f"<p class='note'>Audit status: {'PASS' if audit['ok'] else 'REVIEW'} - "
        f"{len(audit['provenance'])} provenance rows, "
        f"{len(audit['source_coverage'])} source-coverage rows, "
        f"{len(audit['claims'])} audited pitch claims.</p>"
        f"{findings_html}"
        f"<h3>Stored counts and registry status</h3>{table(audit['provenance'])}"
        f"<h3>Source coverage</h3>{table(audit['source_coverage'])}"
    )

    toc = "".join(f'<a href="#{slugify(t)}">{html_lib.escape(t)}</a>' for t in [
        "Recommendation", "Research questions", "The three exhibits",
        "RQ1 \u00b7 response", "RQ2 \u00b7 adoption", "RQ3 \u00b7 valuation",
        "Valuation and the short", "Limitations and disconfirming evidence",
        "Assumption and source ledger", "Registry verification", "Methodology", "Sources",
    ])

    exhibit_note = (
        f"<p class='note'>Each exhibit ships as a PNG in <code>docs/exhibits/</code> for the "
        f"deck. Render with <code>build_site.py</code> or the app's export button.</p>"
        + (f"<p class='note'>Missing: {', '.join(missing)}</p>" if missing else "")
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Short PHVS - Clinical Expectations Lab</title>
<meta name="description" content="Research component of a short thesis on Pharvaris (PHVS):
the clinical, adoption and valuation assumptions the observed price requires, and which of
them have actually been measured.">
<script src="{PLOTLY_JS}"></script>
<style>
:root {{
  --blue: {cf.PHVS_BLUE}; --blue-dark: {cf.PHVS_BLUE_DARK}; --blue-pale: {cf.PHVS_BLUE_PALE};
  --ink: {cf.INK}; --grey: {cf.GREY_600}; --line: {cf.GREY_300}; --bg: {cf.GREY_100};
  --green: {cf.GREEN}; --red: {cf.RED}; --amber: {cf.AMBER};
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#fff; color:var(--ink);
  font-family:"Helvetica Neue", Helvetica, Arial, sans-serif; font-size:15px; line-height:1.6; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:0 24px 80px; }}
header.hero {{ background:var(--blue-dark); color:#fff; padding:44px 0 38px; margin-bottom:28px; }}
header.hero .wrap {{ padding-bottom:0; }}
.kicker {{ font-size:11.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--grey);
  margin-bottom:6px; }}
header.hero .kicker {{ color:#9FC0FF; }}
h1 {{ font-size:34px; line-height:1.2; margin:0 0 10px; }}
h2 {{ font-size:21px; color:var(--blue-dark); margin:0 0 10px; }}
h3 {{ font-size:17px; color:var(--blue-dark); margin:26px 0 8px; }}
h4 {{ font-size:13px; letter-spacing:.09em; text-transform:uppercase; color:var(--grey);
  margin:20px 0 4px; }}
.lead {{ font-size:17px; max-width:78ch; color:#E8EEFB; margin:0; }}
.banner {{ background:#FFF7E6; border:1px solid #B57A00; color:var(--ink); padding:11px 15px;
  border-radius:5px; margin:18px 0 0; font-size:14px; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(215px,1fr)); gap:12px;
  margin:26px 0 8px; }}
.metric {{ background:#fff; border:1px solid var(--line); border-radius:7px; padding:13px 15px; }}
.mlabel {{ font-size:11.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--grey); }}
.mvalue {{ font-size:24px; font-weight:600; color:var(--blue); margin-top:4px; line-height:1.25; }}
.msub {{ font-size:12.5px; color:var(--grey); margin-top:3px; }}
.panel {{ border:1px solid var(--line); border-radius:9px; padding:22px 24px 18px; margin:20px 0;
  background:#fff; }}
.panel figure {{ margin:8px 0 0; }}
.chart {{ border-top:1px solid var(--line); padding-top:6px; margin-top:18px; }}
.chart h3 {{ margin-top:4px; }}
figcaption {{ font-size:13px; color:var(--grey); margin-top:6px; }}
.note {{ font-size:13.5px; color:var(--grey); border-left:3px solid var(--blue-pale);
  padding:5px 11px; margin:8px 0; }}
.headline {{ font-size:17px; font-weight:600; color:var(--blue); margin:0 0 12px; }}
.step {{ max-width:88ch; }}
.rq {{ border-top:1px solid var(--line); padding-top:8px; margin-top:26px; }}
.rq:first-child {{ border-top:0; margin-top:6px; }}
ul.src {{ font-size:12.5px; color:var(--grey); padding-left:18px; margin:6px 0 0; }}
img.exhibit {{ display:block; width:100%; height:auto; border:1px solid var(--line);
  border-radius:8px; margin:6px 0 4px; }}
ol.findings {{ max-width:92ch; }}
ol.findings li {{ margin-bottom:12px; }}
.src {{ font-size:12.5px; color:var(--grey); margin-top:3px; }}
.conclusion {{ background:var(--blue-pale); border:1px solid var(--blue); border-radius:8px;
  padding:20px 22px; font-size:17px; margin:24px 0; }}
.alert {{ background:#FFF7E6; border:1px solid #B57A00; border-radius:6px; padding:14px 16px;
  font-size:14.5px; margin:16px 0; }}
.tablewrap {{ overflow-x:auto; margin-top:10px; }}
table.tbl {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
table.tbl th {{ text-align:left; background:var(--bg); color:var(--blue-dark); font-weight:600;
  padding:8px 10px; border-bottom:2px solid var(--line); position:sticky; top:0; }}
table.tbl td {{ padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
table.tbl tr:hover td {{ background:#F7FAFF; }}
pre {{ background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:14px;
  overflow-x:auto; font-size:13px; }}
code {{ font-family:"SF Mono", Menlo, Consolas, monospace; font-size:.92em;
  background:var(--bg); padding:1px 4px; border-radius:3px; }}
blockquote {{ margin:10px 0; padding:8px 14px; border-left:3px solid var(--blue);
  background:var(--blue-pale); }}
nav.toc {{ display:flex; flex-wrap:wrap; gap:8px; margin:22px 0 4px; }}
nav.toc a {{ font-size:12.5px; text-decoration:none; color:var(--blue-dark);
  border:1px solid var(--line); border-radius:20px; padding:5px 12px; background:#fff; }}
nav.toc a:hover {{ background:var(--blue-pale); border-color:var(--blue); }}
footer {{ border-top:1px solid var(--line); margin-top:36px; padding-top:18px; font-size:13px;
  color:var(--grey); }}
@media (max-width:720px) {{ h1 {{ font-size:26px; }} .panel {{ padding:16px 14px; }} }}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
  <div class="kicker">PHVS Clinical Expectations Lab \u00b7 research component of a short thesis</div>
  <h1>{rec['recommendation']} PHVS at ${price:,.2f}</h1>
  <p class="lead">{_inline(rec['thesis'])}</p>
  <div class="banner">Research for a short thesis, not personalised investment advice. Scenario
  outputs are uncalibrated model results, not forecasts or consensus estimates. Data as of
  {dt.date(2026, 10, 2).isoformat()}.</div>
</div></header>

<div class="wrap">
  <div class="metrics">{metrics}</div>

  <div class="conclusion">{_inline(conclusion)}</div>

  <nav class="toc">{toc}</nav>

  <section class="panel" id="recommendation">
    <div class="kicker">Section 1</div>
    <h2>Recommendation and the findings that carry it</h2>
    <p>{_inline(rec['thesis'])}</p>
    <ol class="findings">{findings}</ol>
    <p class="note">Each finding is a completed calculation on published data. Nothing in this
    list is a forecast, and no input was moved in order to reach the conclusion.</p>
  </section>

  <section class="panel" id="research-questions">
    <div class="kicker">Section 2</div>
    <h2>Three research questions</h2>
    <p class="note">Each question is resolved in the same four steps - methodology, finding,
    investment implication, valuation impact - and carries a status of completed, proposed or
    assumed. Where the data cannot answer, the block returns a break-even threshold.</p>
    {rqs}
    <h3>What is completed, proposed or assumed</h3>
    {table(workstreams)}
  </section>

  <section class="panel" id="the-three-exhibits">
    <div class="kicker">Section 3</div>
    <h2>The three slide-ready exhibits</h2>
    {exhibit_note}
    {exhibit_body}
  </section>

  {''.join(fig_sections)}

  <section class="panel" id="valuation-and-the-short">
    <div class="kicker">Section 6</div>
    <h2>Valuation and the short</h2>
    <p class="note">The pitch claim audit: {len(dl.get_pitch_claim_audit())} claims traced to
    the sources behind them. Findings are listed, never silently corrected.</p>
    {table(dl.get_pitch_claim_audit())}
    {table(pd.DataFrame(repro["findings"]))}
    <p class="note">Required equity value {usdf(bundle['required_equity_value'])} against a
    modelled {usdf(bundle['stated_equity_value'])}; gap {usdf(bundle['gap_equity'])}.</p>
    <h3>Scenarios</h3>
    {table(scenarios)}
    <h3>Short scenarios, net of carry</h3>
    {table(val.short_table(base))}
  </section>

  <section class="panel" id="limitations-and-disconfirming-evidence">
    <div class="kicker">Section 7</div>
    <h2>Disconfirming evidence and missing evidence</h2>
    <h3>The strongest case against this short, at full strength</h3>
    {disconfirming}
    <h3>Where the lab itself is weakest</h3>
    {weakest}
    <h3>Evidence still missing</h3>
    {table(invalid)}
  </section>

  <section class="panel" id="assumption-and-source-ledger">
    <div class="kicker">Ledger</div>
    <h2>Assumption and source ledger</h2>
    <p class="note">{len(assumptions)} parameters. Every row carries its arithmetic type
    (reported / calculated / assumed), verification status and source.</p>
    {table(viz.format_assumption_ledger(assumptions))}
  </section>

  <section class="panel" id="registry-verification">
    <div class="kicker">Audit</div>
    <h2>Registry verification</h2>
    <p class="note">Stored counts re-checked against the primary registry record. Failures
    degrade to <code>not_found_in_registry</code>; nothing is ever invented. ClinicalTrials.gov
    holds no results record for NCT06669754, so the CHAPTER-3 threshold rows remain
    company-reported and provisional.</p>
    {audit_html}
  </section>

  <section class="panel" id="methodology">
    <div class="kicker">Method</div>
    <h2>Methodology</h2>
    {md_to_html((REPO / "METHODOLOGY.md").read_text())}
  </section>

  <section class="panel" id="sources">
    <div class="kicker">Provenance</div>
    <h2>Sources</h2>
    {table(dl.get_sources())}
  </section>

  <section class="panel">
    <div class="kicker">Readme</div>
    <h2>Repository</h2>
    {md_to_html((REPO / "README.md").read_text())}
  </section>

  <footer>
    <p>Built by <code>build_site.py</code> from <code>phvs_lab.modules</code> - the same
    functions the Streamlit app calls, so the published numbers cannot drift from the app.
    Source: <a href="https://github.com/maanas-pab/phvs-clinical-expectations-lab">github.com/maanas-pab/phvs-clinical-expectations-lab</a>.</p>
    <p>Research for a short thesis. Not investment advice. Not a forecast. No patient-level
    data, physician interviews or consensus estimates are generated anywhere in this project.</p>
  </footer>
</div>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
