"""Build the static GitHub Pages site from the same modules the app uses.

Every figure and number here is produced by ``phvs_lab.modules`` - the same
functions Streamlit calls - so the published page cannot drift from the app.

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
from phvs_lab.modules import response_engine as re_
from phvs_lab.modules import trial_audit as ta
from phvs_lab.modules import valuation as val
from phvs_lab.utils import visualization as viz

REPO = Path(__file__).resolve().parent
OUT = REPO / "docs" / "index.html"

PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# --------------------------------------------------------------------------
# Markdown -> HTML (headings, tables, lists, code, emphasis)
# --------------------------------------------------------------------------

def _inline(text: str) -> str:
    text = html_lib.escape(text, quote=False)
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

def chart(fig, caption: str = "") -> str:
    fig.update_layout(**{k: v for k, v in cf.PLOT_LAYOUT.items() if k in ("font", "legend")},
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


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build() -> Path:
    base = val.params_from_assumptions()
    price = base.stock_price
    trials = dl.get_clinical_trials()
    repro = val.reproduce_pitch_valuation(dl.get_valuation_model())
    scenarios = val.scenario_table(base)
    summary = val.value_summary(base)
    sens = val.sensitivity_ranking(base)
    impl = val.market_implied(base)
    bundle = val.required_bundle(base)
    short = val.short_table(base)
    ledger = ec.population_ledger()
    params = ec.summarize_economics()["params"]
    dist = list(re_.distributions_for_trial("CHAPTER-3", trials).values())[0]
    window_curves = ta.window_sensitivity_curves([0.4, 0.8], [0.3, 0.5, 1.0],
                                                 [30, 90, 168, 182, 365])
    adjusted = ta.window_adjusted_table(trials, target_days=180, dispersion=0.5)
    surface = pm.value_surface(np.linspace(0.6, 1.5, 19), np.linspace(0.02, 0.50, 19))
    assumptions = dl.get_commercial_assumptions()
    targets = impl.set_index("input")

    req_peak = targets.loc["peak_penetration_prophylaxis", "required_value"]
    req_approval = targets.loc["approval_prophylaxis", "required_value"]
    req_discount = targets.loc["discount_rate", "required_value"]

    def pctf(x, d=1):
        return f"{x * 100:.{d}f}%"

    def usdf(x):
        return f"${x:,.0f}"

    metrics = "".join([
        metric("Observed price (2 Oct 2026 close)", f"${price:,.2f}", "NASDAQ: PHVS"),
        metric("Pitch enterprise value (reproduced)", usdf(repro["enterprise_value"]),
               f"the price needs {repro['multiple_required']:.2f}× that"),
        metric("Model value per share", f"${summary['value_per_share']:,.2f}",
               f"{summary['upside_downside_pct']:+.1%} vs the observed price"),
        metric("Equity value the price requires", usdf(bundle["required_equity_value"]),
               f"model {usdf(bundle['stated_equity_value'])}"),
        metric("Required peak penetration", pctf(req_peak),
               f"stated {pctf(base.peak_penetration_prophylaxis)}"),
        metric("Required approval probability", pctf(req_approval),
               f"stated {pctf(base.approval_prophylaxis)}"),
        metric("Required discount rate", pctf(req_discount),
               f"stated {pctf(base.discount_rate)}"),
        metric("Bear / bull per share",
               f"${scenarios.set_index('scenario').loc['bear', 'value_per_share']:,.2f} / "
               f"${scenarios.set_index('scenario').loc['bull', 'value_per_share']:,.2f}",
               "scenarios, not forecasts"),
    ])

    conclusion = cf.CONCLUSION_TEMPLATE.format(
        X=f"{usdf(bundle['required_enterprise_value'])} of enterprise value - "
          f"{repro['multiple_required']:.2f}× the reproduced pitch model and "
          f"{bundle['required_multiple_of_stated_ev']:.2f}× the extended model",
        Y=f"${summary['value_per_share']:,.2f} per share on the stated inputs, against "
          f"${repro['value_per_share_current_shares']:,.2f} on the pitch's own cash flows",
        Z="the real net price, the real switching rate, durability beyond 24 weeks "
          "and the subtype split behind the attack-free count",
    )

    figures = [
        ("The pitch, reproduced first",
         chart(viz.fig_pitch_vs_models(repro, scenarios, price),
               "The pitch's own cash flows re-run on its terms, then on the verified 70.2m "
               "share count and verified net cash, against the observed price."),
         ["Share count, terminal omission and embedded-price findings are listed, not silently "
          "corrected."]),
        ("Equity bridge",
         chart(viz.fig_equity_bridge(summary),
               "Enterprise value plus verified net cash of "
               f"{usdf(base.net_cash)} equals equity value; per share on 70.2m shares."),
         []),
        ("Present value by year",
         chart(viz.fig_pv_profile(val.build_cashflows(base)),
               "Fixed costs at face value; product contributions weighted once by approval "
               "probability; terminal growth forced to zero after the loss-of-exclusivity year."),
         []),
        ("What the price requires, one input at a time",
         chart(viz.fig_required_inputs(impl, sens),
               "Each input solved separately so the model equals the observed price. The price "
               "is a bundle, so these cannot all be true at once."),
         [f"Peak penetration {pctf(req_peak)} against {pctf(base.peak_penetration_prophylaxis)} "
          f"stated; approval {pctf(req_approval)} against {pctf(base.approval_prophylaxis)}; "
          f"discount {pctf(req_discount)} against {pctf(base.discount_rate)}."]),
        ("One input shocked at a time",
         chart(viz.fig_tornado(sens, summary["value_per_share"]),
               "Sensitivity of value per share to a single shock, all else held at the "
               "reference case."),
         []),
        ("Chapter 3 response reconstruction",
         chart(viz.fig_response_categories(dist), dist.window_note()),
         ["Exclusive categories are subtraction of the published cumulative thresholds; they are "
          "only point identified when all four thresholds are disclosed."]),
        ("Exact intervals on the disclosed thresholds",
         chart(viz.fig_response_thresholds(dist.proportion_intervals()),
               "Clopper-Pearson intervals: sampling uncertainty on the observed counts, stated "
               "separately from window length."),
         []),
        ("Window mechanics",
         chart(viz.fig_window_sensitivity(window_curves),
               "Attack-free is a count over a stated window and falls mechanically as the "
               "window lengthens at an unchanged rate."),
         []),
        ("Window-standardised comparison",
         chart(viz.fig_window_adjustment(adjusted),
               "A gamma-Poisson standardisation to a common window. This is a stated "
               "transformation of published counts, not a re-analysis and not a randomised "
               "comparison between separate trials."),
         []),
        ("Efficacy to breakthrough cases",
         chart(viz.fig_efficacy_projection(
             ec.efficacy_projection(params, np.linspace(0.40, 1.00, 25)),
         ), f"Dispersion k={ec.bridge_components(params)['fitted_dispersion_k']:.4g} calibrated "
            "once from the trial's published values, then held fixed while efficacy varies."),
         []),
        ("Population funnel",
         chart(viz.fig_population_funnel(ledger),
               "Segments are mutually exclusive and validated to sum to 1.0 of the eligible "
               "pool."),
         []),
        ("Prevention and rescue",
         chart(viz.fig_prevention_ladder(ec.prevention_rescue_ladder(params)),
               "Attacks, patients and revenue are three separate quantities with their own "
               "conversion assumptions."),
         []),
        ("Prescribing map to value per share",
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
        ("Choice shares by segment",
         chart(viz.fig_choice_shares(pm.choice_probabilities()),
               "Stay on current prophylaxis, switch to an existing oral, switch to PHVS oral. "
               "Coefficients are ledger assumptions, not measured preferences."),
         []),
        ("Scenarios and the short",
         chart(viz.fig_short_scenarios(short),
               "Net of a stated annual borrow cost over the stated horizon. Not a "
               "recommendation."),
         []),
        ("Evidence coverage",
         chart(viz.fig_evidence_coverage(ev.provenance_summary(trials)),
               "Every observation is tagged with its source and verification status."),
         []),
    ]

    figures_html = ""
    for i, (title, fig_html, notes) in enumerate(figures, start=1):
        notes_html = "".join(f"<p class='note'>{_inline(n)}</p>" for n in notes)
        figures_html += (f"<section class='panel'><div class='kicker'>Chart {i:02d}</div>"
                         f"<h2>{html_lib.escape(title)}</h2>{notes_html}{fig_html}</section>")

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
    invalid = val.invalidation_conditions(base)

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PHVS Clinical Expectations Lab</title>
<meta name="description" content="Reverse-engineering the clinical and prescribing assumptions
 required to justify the observed Pharvaris share price.">
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
figcaption {{ font-size:13px; color:var(--grey); margin-top:6px; }}
.note {{ font-size:13.5px; color:var(--grey); border-left:3px solid var(--blue-pale);
  padding:5px 11px; margin:8px 0; }}
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
  <div class="kicker">PHVS Clinical Expectations Lab</div>
  <h1>What would have to be true for Pharvaris at ${price:,.2f}</h1>
  <p class="lead">A measurement instrument, not a forecast: the clinical, prescribing and
  commercial assumptions required to justify the observed share price, each tagged with its
  source and verification status - and the ones nobody has measured yet.</p>
  <div class="banner">Not investment advice. Scenario outputs are uncalibrated model results,
  not forecasts or consensus estimates. Data as of {dt.date(2026, 10, 2).isoformat()}.</div>
</div></header>

<div class="wrap">
  <div class="metrics">{metrics}</div>

  <div class="conclusion">{_inline(conclusion)}</div>

  <nav class="toc">
    <a href="#pitch">Pitch reproduction</a><a href="#what-the-price-requires-one-input-at-a-time">Required inputs</a>
    <a href="#chapter-3-response-reconstruction">Response reconstruction</a>
    <a href="#window-mechanics">Window mechanics</a>
    <a href="#population-funnel">Population</a>
    <a href="#prescribing-map-to-value-per-share">Prescribing map</a>
    <a href="#scenarios-and-the-short">Scenarios</a>
    <a href="#ledger">Ledger</a><a href="#audit">Audit</a><a href="#invalidation">Invalidation</a>
    <a href="#methodology">Methodology</a><a href="#sources">Sources</a>
  </nav>

  <section class="panel" id="pitch">
    <div class="kicker">Section 1</div>
    <h2>The pitch claim audit</h2>
    <p class="note">{len(dl.get_pitch_claim_audit())} pitch claims traced to the sources behind
    them. Findings are listed, never silently corrected.</p>
    {table(dl.get_pitch_claim_audit())}
    {table(pd.DataFrame(repro["findings"]))}
  </section>

  {figures_html}

  <section class="panel" id="ledger">
    <div class="kicker">Ledger</div>
    <h2>Assumption and source ledger</h2>
    <p class="note">{len(assumptions)} parameters. Every row carries its arithmetic type
    (reported / calculated / assumed), verification status and source.</p>
    {table(viz.format_assumption_ledger(assumptions))}
  </section>

  <section class="panel" id="audit">
    <div class="kicker">Audit</div>
    <h2>Registry verification</h2>
    <p class="note">Stored counts re-checked against the primary registry record. Failures
    degrade to <code>not_found_in_registry</code>; nothing is ever invented.</p>
    {audit_html}
  </section>

  <section class="panel" id="invalidation">
    <div class="kicker">Evidence gaps</div>
    <h2>What would invalidate this</h2>
    {table(invalid)}
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
    <p>Not investment advice. Not a forecast. No patient-level data, physician interviews or
    consensus estimates are generated anywhere in this project.</p>
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
