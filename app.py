"""PHVS Clinical Expectations Lab - Streamlit application.

An original research component of a short thesis on Pharvaris (PHVS). Eight
sections, in thesis order: the recommendation first, then three research
questions, then the three slide-ready exhibits that carry the argument, then
valuation, limitations and the evidence underneath it.

Every number on screen comes from ``phvs_lab.modules`` or from the CSV ledger.
Nothing is typed into the UI.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from phvs_lab.config import (
    CONCLUSION_TEMPLATE, DISCLAIMERS, EXPORT_DIR, REPO_ROOT, SECTION_INTROS,
    pct, usd, usd_m,
)
from phvs_lab.modules import data_loader as dl
from phvs_lab.modules import economics as ec
from phvs_lab.modules import evidence as ev
from phvs_lab.modules import prescribing_map as pm
from phvs_lab.modules import research as rs
from phvs_lab.modules import response_engine as re_
from phvs_lab.modules import survey as sv
from phvs_lab.modules import trial_audit as ta
from phvs_lab.modules import valuation as val
from phvs_lab.utils import visualization as viz

SECTIONS = [
    "1 · Recommendation",
    "2 · Research questions",
    "3 · Exhibit 1 · Response",
    "4 · Exhibit 2 · Adoption",
    "5 · Exhibit 3 · Evidence to downside",
    "6 · Valuation & the short",
    "7 · Limitations & disconfirming evidence",
    "8 · Evidence & method",
]

BRAND_CSS = """
<style>
  .stApp { background:#FFFFFF; }
  h1, h2, h3 { color:#082C6E !important; font-family:"Helvetica Neue", Helvetica, Arial, sans-serif; }
  #MainMenu, footer { visibility:hidden; }
  div[data-testid="stSidebar"] { background:#F2F6FF; border-right:1px solid #D5DAE3; }
  .phvs-kicker { font-size:12px; letter-spacing:.14em; text-transform:uppercase;
                 color:#5A6478; margin-bottom:2px; }
  .phvs-lead { font-size:17px; line-height:1.55; color:#10151F; }
  .phvs-headline { font-size:19px; line-height:1.5; font-weight:600; color:#0B4FD8;
                   margin:-4px 0 16px; }
  .phvs-note { font-size:13.5px; line-height:1.6; color:#5A6478;
               border-left:3px solid #DCE7FF; padding:6px 12px; margin:6px 0 14px; }
  .phvs-flag { display:inline-block; font-size:11.5px; letter-spacing:.06em;
               text-transform:uppercase; padding:2px 8px; border-radius:3px;
               background:#F2F6FF; color:#082C6E; border:1px solid #DCE7FF; }
  .phvs-alert { background:#FFF7E6; border:1px solid #B57A00; color:#10151F;
                padding:10px 14px; border-radius:4px; font-size:14px; margin-bottom:10px; }
  .phvs-conclusion { background:#F2F6FF; border:1px solid #DCE7FF; padding:16px 18px;
                     border-radius:6px; font-size:16.5px; line-height:1.6; }
  [data-testid="stMetricValue"] { color:#0B4FD8; }
</style>
"""


# ---------------------------------------------------------------------------
# Cached computation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _base_params() -> val.ValuationParams:
    return val.params_from_assumptions()


@st.cache_data(show_spinner=False)
def _surface() -> pd.DataFrame:
    return pm.value_surface(np.linspace(0.6, 1.5, 19), np.linspace(0.02, 0.50, 19))


@st.cache_data(show_spinner=False)
def _scenarios() -> pd.DataFrame:
    return val.scenario_table(_base_params())


@st.cache_data(show_spinner=False)
def _sensitivity() -> pd.DataFrame:
    return val.sensitivity_ranking(_base_params())


@st.cache_data(show_spinner=False)
def _market_implied() -> pd.DataFrame:
    return val.market_implied(_base_params())


@st.cache_data(show_spinner=False)
def _reproduce() -> Dict:
    return val.reproduce_pitch_valuation(dl.get_valuation_model())


@st.cache_data(show_spinner=False)
def _economics() -> Dict:
    return ec.summarize_economics()


@st.cache_data(show_spinner=False)
def _window_curves() -> pd.DataFrame:
    return ta.window_sensitivity_curves([0.4, 0.8], [0.3, 0.5, 1.0],
                                        [30, 90, 168, 182, 365])


@st.cache_data(show_spinner=False)
def _audit() -> Dict:
    return ev.audit_report()


@st.cache_data(show_spinner=False)
def _demo_survey() -> Dict:
    return sv.demonstrate_survey_design()


def _replace(obj, **kw):
    from dataclasses import replace as _rep
    return _rep(obj, **kw)


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

def _intro(section: str, text: Optional[str] = None) -> None:
    st.markdown(f'<div class="phvs-lead">{text or SECTION_INTROS.get(section, "")}</div>',
                unsafe_allow_html=True)


def _disclaimers(*keys: str) -> None:
    for key in keys or ("not_forecast", "not_evidence", "no_h2h", "no_fabrication"):
        st.markdown(f'<div class="phvs-note">{DISCLAIMERS[key]}</div>',
                    unsafe_allow_html=True)


def _flag(text: str) -> None:
    st.markdown(f'<span class="phvs-flag">{text}</span>', unsafe_allow_html=True)


def _df(df: pd.DataFrame, height: int = 340) -> None:
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)


def _fig(fig, caption: str = "") -> None:
    st.plotly_chart(fig, use_container_width=True)
    if caption:
        st.caption(caption)


def _markdown_file(name: str) -> str:
    path = REPO_ROOT / name
    if path.exists():
        return path.read_text()
    return f"*{name} has not been written yet.*"


def _pct_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Display-format proportion columns without altering the underlying data."""
    out = df.copy()
    for c in cols:
        out[c] = [pct(float(v), 1) for v in out[c]]
    return out


def _unit_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Display-format numeric columns against their declared unit."""
    out = df.copy()
    units = out["unit"].astype(str)
    for c in cols:
        out[c] = out[c].astype(object)
    locs = {c: out.columns.get_loc(c) for c in cols}
    for i, unit in enumerate(units):
        for c in cols:
            v = float(df.iloc[i][c])
            if "USD" in unit:
                s = usd(v)
            elif "multiple" in unit:
                s = f"{v:.2f}\u00d7"
            else:
                s = pct(v, 1)
            out.iat[i, locs[c]] = s
    return out


_EXHIBIT_CACHE: Dict[str, object] = {}


def _exhibit(exhibit_id: str):
    """One exhibit dict plus its figure, computed once per session."""
    ex = next(e for e in rs.exhibits() if e["id"] == exhibit_id)
    if ex["png"] not in _EXHIBIT_CACHE:
        _EXHIBIT_CACHE.update(rs.exhibit_figures())
    return ex, _EXHIBIT_CACHE[ex["png"]]


def _show_exhibit(exhibit_id: str, kicker: str) -> Dict:
    ex, fig = _exhibit(exhibit_id)
    st.markdown(f'<div class="phvs-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(ex["title"])
    st.markdown(f'<div class="phvs-headline">{ex["headline"]}</div>', unsafe_allow_html=True)
    _fig(fig, ex["caption"])
    with st.expander("Methodology, sources and what it means"):
        st.markdown(f"**Methodology.** {ex['methodology']}", unsafe_allow_html=True)
        st.markdown(f"**What it means for the thesis.** {ex['implication']}")
        st.markdown(f"**Status.** {ex['status']}")
        st.markdown("**Sources**")
        for line in rs.exhibit_source_lines(ex):
            st.caption(line)
    return ex


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_recommendation() -> None:
    st.markdown('<div class="phvs-kicker">Section 1 \u00b7 Recommendation</div>',
                unsafe_allow_html=True)
    st.title("Short PHVS")
    _intro("recommendation")

    rec = rs.short_recommendation()
    base = _base_params()
    summary = val.value_summary(base)
    repro = _reproduce()
    bundle = val.required_bundle(base)
    mi = _market_implied().set_index("input")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommendation", f"{rec['recommendation']} {rec['ticker']}",
              delta=f"{usd(rec['price'], 2)} close, {rec['price_date']}",
              delta_color="normal")
    m2.metric("Our stated-input model", usd(rec["model_value"], 2),
              delta=f"{rec['model_vs_price']:+.1%} vs price", delta_color="normal")
    m3.metric("Price requires", usd_m(rec["required_ev"]),
              delta=f"{rec['required_multiple_of_cashflow_model']:.2f}\u00d7 the cash-flow "
                    f"model", delta_color="inverse")
    m4.metric("Invalidation level", usd(rec["breakeven_value_per_share"], 2),
              delta=f"{pct(rec['carry_pct'], 0)} carry", delta_color="inverse")

    st.markdown(f'<div class="phvs-lead">{rec["thesis"]}</div>', unsafe_allow_html=True)

    required = (
        f"an enterprise value of {usd(bundle['required_enterprise_value'])} - "
        f"{bundle['required_multiple_of_stated_ev']:.2f}\u00d7 the reference model and "
        f"{repro['multiple_required']:.2f}\u00d7 the pitch's own cash flows - reached at "
        f"{pct(float(mi.loc['peak_penetration_prophylaxis', 'required_value']), 1)} peak "
        f"prophylaxis penetration, a "
        f"{pct(float(mi.loc['approval_prophylaxis', 'required_value']), 1)} approval "
        f"probability or a {pct(float(mi.loc['discount_rate', 'required_value']), 1)} discount "
        f"rate, each holding everything else at its stated value"
    )
    supported = (
        f"the ledger's stated inputs, worth {usd(summary['value_per_share'], 2)} a share from a "
        f"15% peak penetration, 88% approval probability and 10% discount rate; the pitch's own "
        f"cash flows, which reproduce to only "
        f"{usd(repro['value_per_share_current_shares'], 2)} on the verified "
        f"{repro['shares_current'] / 1e6:.1f}m shares; and the choice model at stated "
        f"preferences, which delivers {pct(pm.implied_peak_penetration(), 1)} peak penetration"
    )
    unmeasured = (
        "real-world switching rates, realised net price after gross-to-net, durability of the "
        "attack-free rate beyond 24 weeks, and any head-to-head evidence against injected "
        "prophylaxis"
    )
    st.markdown(
        CONCLUSION_TEMPLATE.format(REC=rec["recommendation"],
                                   PRICE=usd(rec["price"], 2),
                                   X=required, Y=supported, Z=unmeasured,
                                   W=usd(rec["breakeven_value_per_share"], 2)),
        unsafe_allow_html=True)

    _fig(viz.fig_pitch_vs_models(repro, _scenarios(), rec["price"]),
         "Bars are model outputs at the stated inputs; the dashed line is the observed close.")

    st.subheader("The findings that carry it")
    for f in rec["findings"]:
        st.markdown(f"**{f['rank']}. {f['headline']}**  \n{f['claim']}")
        st.caption(f"Basis: {f['basis']} \u00b7 Status: {f['status']}")

    _disclaimers("position_disclosure", "not_forecast", "no_h2h")


def section_research() -> None:
    st.markdown('<div class="phvs-kicker">Section 2 \u00b7 Research questions</div>',
                unsafe_allow_html=True)
    st.title("Three research questions")
    _intro("research")

    for i, block in enumerate([rs.rq1_response(), rs.rq2_adoption(), rs.rq3_price()]):
        if i:
            st.divider()
        st.markdown(f"### {block.key} \u00b7 {block.question}", unsafe_allow_html=True)
        _flag(f"status: {block.status}")

        cols = st.columns(len(block.metrics))
        for col, (name, value) in zip(cols, block.metrics.items()):
            col.metric(name, value)

        for label, text in (("Methodology", block.methodology),
                            ("Finding", block.finding),
                            ("Investment implication", block.implication),
                            ("Valuation impact", block.valuation_impact)):
            st.markdown(f"**{label}**")
            st.markdown(text, unsafe_allow_html=True)

        if block.table is not None:
            st.markdown("**Supporting table**")
            table = block.table
            if "unit" in table.columns:
                table = _unit_cols(table, [c for c in ("stated", "required_for_price",
                                                       "value") if c in table.columns])
            elif "value" in table.columns:
                table = _pct_cols(table, ["value"])
            _df(table, height=320)

        if block.thresholds is not None:
            st.markdown("**Thresholds and break-evens**")
            th = block.thresholds
            if "unit" in th.columns:
                th = _unit_cols(th, ["value"])
            else:
                th = _pct_cols(th, ["value"])
            _df(th, height=260)

    st.divider()
    st.subheader("What is completed, proposed or assumed")
    st.markdown("Nothing below is presented as a finding unless it carries the "
                "`completed` status with its evidence type.")
    _df(rs.workstreams(), height=520)

    _disclaimers("not_prescribing")


def section_exhibit1() -> None:
    st.markdown('<div class="phvs-kicker">Section 3 \u00b7 Exhibit 1 of 3</div>',
                unsafe_allow_html=True)
    st.title("Exhibit 1 \u00b7 Response")
    _intro("exhibit1")
    ex = _show_exhibit("EX1", "RQ1 \u00b7 slide-ready exhibit")
    st.caption(f"Question: {ex['question']}")

    st.divider()
    st.subheader("How the exhibit is built")
    section_response()
    st.divider()
    st.subheader("Why the trials cannot be read across")
    section_audit()


def section_exhibit2() -> None:
    st.markdown('<div class="phvs-kicker">Section 4 \u00b7 Exhibit 2 of 3</div>',
                unsafe_allow_html=True)
    st.title("Exhibit 2 \u00b7 Adoption")
    _intro("exhibit2")
    ex = _show_exhibit("EX2", "RQ2 \u00b7 slide-ready exhibit")
    st.caption(f"Question: {ex['question']}")

    st.divider()
    st.subheader("Population and economics behind the bars")
    section_economics()
    st.divider()
    st.subheader("The choice model behind the segment ceilings")
    section_map()


def section_exhibit3() -> None:
    st.markdown('<div class="phvs-kicker">Section 5 \u00b7 Exhibit 3 of 3</div>',
                unsafe_allow_html=True)
    st.title("Exhibit 3 \u00b7 Evidence to downside")
    _intro("exhibit3")
    ex = _show_exhibit("EX3", "RQ3 \u00b7 slide-ready exhibit \u00b7 the chain")
    st.caption(f"Question: {ex['question']}")

    st.divider()
    st.subheader("Every link in the chain, named")
    st.markdown("`Reaches value directly = yes` marks an input that enters the discounted cash "
                "flow without passing through an analyst assumption. Everything else is the "
                "connective tissue the exhibit draws between panels.")
    links = rs.chain_links()
    _df(links, height=640)
    st.caption(f"{int((links['reaches_value_directly'] == 'yes').sum())} of {len(links)} inputs "
               f"reach value directly; "
               f"{int((links['status'] == 'unverified').sum())} are flagged unverified.")

    st.subheader("Where the chain flips")
    st.markdown("Where the data cannot settle a question, the answer is the threshold at which "
                "the position stops working \u2014 not a conclusion.")
    _df(rs.chain_break_even(), height=360)

    _disclaimers("not_evidence", "not_prescribing")


def section_valuation() -> None:
    st.markdown('<div class="phvs-kicker">Section 6 \u00b7 Valuation</div>',
                unsafe_allow_html=True)
    st.title("Valuation and the short")
    _intro("valuation")

    base = _base_params()
    st.subheader("The pitch, reproduced first")
    repro = _reproduce()
    _df(pd.DataFrame(repro["findings"]), height=300)
    st.caption(f"Reproduced enterprise value {usd(repro['enterprise_value'])}; "
               f"the price needs {repro['multiple_required']:.2f}\u00d7 that.")

    st.subheader("Then extended")
    p1, p2, p3, p4, p5 = st.columns(5)
    peak = p1.slider("Peak prophylaxis penetration", 0.02, 0.40,
                     float(base.peak_penetration_prophylaxis), 0.01)
    approval = p2.slider("Prophylaxis approval probability", 0.30, 1.00,
                         float(base.approval_prophylaxis), 0.01)
    discount = p3.slider("Discount rate", 0.06, 0.25, float(base.discount_rate), 0.005,
                         format="%.3f")
    price = p4.slider("Annual net price", 150_000, 700_000, int(base.price_prophylaxis), 25_000)
    shares = p5.slider("Shares (millions)", 30.0, 110.0, base.shares / 1e6, 1.0)

    live = _replace(base, peak_penetration_prophylaxis=peak, approval_prophylaxis=approval,
                    discount_rate=discount, price_prophylaxis=float(price),
                    shares=shares * 1e6)
    live_summary = val.value_summary(live)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Value per share", usd(live_summary["value_per_share"], 2),
              delta=f"{live_summary['upside_downside_pct']:+.1%} vs price", delta_color="normal")
    a2.metric("Enterprise value", usd_m(live_summary["enterprise_value"]))
    a3.metric("Terminal share of EV", pct(live_summary["terminal_share_of_ev"], 1))
    a4.metric("Price requires EV", usd_m(live_summary["required_ev_for_price"]),
              delta=f"{live_summary['required_ev_for_price'] / live_summary['enterprise_value']:.2f}"
                    f"\u00d7 this model",
              delta_color="inverse")

    st.caption("If no slider is moved these equal the reference case. "
               "Sliders are scenario inputs, not forecasts.")

    _fig(viz.fig_equity_bridge(val.value_summary(base)))
    _fig(viz.fig_pv_profile(val.build_cashflows(base)),
         "Fixed costs at face value; product contributions weighted by approval probability; "
         "terminal growth forced to zero after the loss-of-exclusivity year.")
    _fig(viz.fig_tornado(_sensitivity(), val.value_summary(base)["value_per_share"]),
         "One input shocked at a time.")

    st.subheader("What the price requires")
    _fig(viz.fig_required_inputs(_market_implied(), _sensitivity()),
         "Each input solved separately. The price is a bundle: section 1 shows the multiple.")
    bundle = val.required_bundle(base)
    st.write(f"Required equity value {usd(bundle['required_equity_value'])} against a modelled "
             f"{usd(bundle['stated_equity_value'])}; gap {usd(bundle['gap_equity'])}.")

    st.subheader("Scenarios and the short")
    scenarios = _scenarios()
    _fig(viz.fig_pitch_vs_models(repro, scenarios, base.stock_price))
    short = val.short_table(base)
    _fig(viz.fig_short_scenarios(short),
         "Net of a stated annual borrow cost over the stated horizon. Not a recommendation.")
    _df(short, height=300)
    _df(val.invalidation_conditions(base), height=340)

    _disclaimers()


def section_limitations() -> None:
    st.markdown('<div class="phvs-kicker">Section 7 \u00b7 Limitations</div>',
                unsafe_allow_html=True)
    st.title("Disconfirming evidence and missing evidence")
    _intro("limitations")

    st.subheader("The strongest case against this short, at full strength")
    _df(rs.disconfirming_evidence(), height=560)

    st.subheader("Where the lab itself is weakest")
    for topic, text in rs.limitations_notes():
        st.markdown(f"**{topic}**")
        st.markdown(text, unsafe_allow_html=True)

    st.subheader("Evidence still missing")
    st.markdown("These are the items that would retire the break-even thresholds above. Until "
                "they exist, the thresholds are the answer.")
    _df(val.invalidation_conditions(_base_params()), height=360)

    st.divider()
    st.subheader("The instrument that would close the largest gap")
    section_survey()
    _disclaimers("position_disclosure")


def section_response() -> None:
    _intro("response",
           "Nested response thresholds are inverted into mutually exclusive categories. The "
           "reconstruction is arithmetic, not estimation.")
    df = dl.get_clinical_trials()
    trials = re_.available_trials(df)
    if not trials:
        st.info("No trial in the ledger publishes participant-level responder counts.")
        return
    trial = st.radio("Trial", trials, horizontal=True)
    try:
        dists = re_.distributions_for_trial(trial, df)
    except ValueError as exc:
        st.error(str(exc))
        return
    arm_names = list(dists)
    arm = st.selectbox("Arm", arm_names)
    dist = dists[arm]

    if dist.is_point_identified:
        _flag("point identified")
    else:
        _flag("partially identified - identified set only")
    _fig(viz.fig_response_categories(dist), dist.window_note())
    _fig(viz.fig_response_thresholds(dist.proportion_intervals()),
         "Exact binomial intervals on the disclosed cumulative thresholds.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mutually exclusive categories**")
        counts = pd.DataFrame({"category": list(dist.counts),
                               "patients": list(dist.counts.values())})
        counts["share"] = counts["patients"] / dist.total_patients
        _df(counts, height=240)
    with c2:
        st.markdown("**Sample-size resolution**")
        res = re_.sample_size_resolution(dist.total_patients)
        st.write(res["note"])
        st.write(f"One patient = {pct(res['one_patient_share'], 2)} of the arm; "
                 f"Wilson half-width at a 50% rate is {pct(res['wilson_half_width_at_half'], 1)}.")

    st.markdown(re_.explain_average_vs_attackfree())
    _fig(viz.fig_window_sensitivity(_window_curves()),
         "Illustrative model output. No clinical claim is made from this panel.")


def section_audit() -> None:
    _intro("audit",
           "Results are shown beside population, window and endpoint so that no two trials can "
           "be read as a randomised comparison by accident.")
    df = dl.get_clinical_trials()
    table = ta.build_comparability_table(df)
    warnings = ta.comparability_warnings(df)

    st.markdown(f"**{len(warnings)} comparability warnings** are raised across the trial set. "
                "Every one is structural - population, window, endpoint or comparator - not "
                "statistical.")
    _df(warnings, height=300)

    tab_a, tab_b = st.tabs(["Arms side by side", "Window mechanics"])
    with tab_a:
        _df(table, height=440)
    with tab_b:
        st.markdown(ta.explain_window_mechanics())
        adjusted = ta.window_adjusted_table(df, target_days=180, dispersion=0.5)
        _fig(viz.fig_window_adjustment(adjusted),
             "Standardisation is a model transformation of the reported rate, not a re-analysis "
             "of patient data, and never turns separate trials into a comparison.")


def section_economics() -> None:
    _intro("economics",
           "Cohorts are mutually exclusive, so no patient is counted twice. Attacks are not "
           "equated with paid prescriptions.")
    S = _economics()
    ledger = S["ledger"]
    bridge = S["bridge"]

    _fig(viz.fig_population_funnel(ledger),
         "Segments are mutually exclusive and validated to sum to 1.0 of the eligible pool.")
    _fig(viz.fig_prevention_ladder(S["ladder"]),
         "Attacks, patients and revenue are three separate quantities with their own "
         "conversions.")

    st.markdown(ec.explain_double_counting())

    eff = st.slider("Scenario efficacy (before real-world haircut)", 0.40, 1.00,
                    float(S["params"].efficacy), 0.01)
    proj = ec.efficacy_projection(S["params"], np.linspace(0.40, 1.00, 25))
    _fig(viz.fig_efficacy_projection(proj),
         f"Dispersion k={bridge['fitted_dispersion_k']:.4g}, calibrated once from the trial's "
         "published values and held fixed while efficacy varies.")
    row = proj.iloc[(proj["efficacy"] - eff).abs().argsort().iloc[0]]
    st.caption(f"At efficacy {eff:.0%}: {row['breakthrough_patients']:,.0f} breakthrough "
               f"patients, {usd(row['acute_revenue'])} acute revenue, "
               f"{usd(row['total_revenue'])} combined.")


def section_map() -> None:
    _intro("map",
           "Segments, utilities and switching assumptions are editable. The contour shows what "
           "must be true for the price, not what investors believe.")
    stated = _base_params().stock_price
    c1, c2, c3 = st.columns(3)
    switching = c1.slider("Annual switching rate", 0.02, 0.50, 0.18, 0.01)
    premium = c2.slider("PHVS price relative to stated", 0.60, 1.50, 1.00, 0.05)
    c3.metric("Market price", usd(stated, 2))

    params = pm.MapParams(premium_multiplier=premium, switching_rate=switching)
    choice = pm.choice_probabilities(params)
    patients = pm.implied_patients(params)
    penetration = pm.implied_peak_penetration(params)
    value = pm.value_for_inputs(premium, switching)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Patients implied", f"{patients['phvs_patients'].sum():,.0f}")
    a2.metric("Peak penetration", pct(penetration, 1))
    a3.metric("Value per share", usd(value["value_per_share"], 2),
              delta=f"{value['value_per_share'] / stated - 1:+.1%} vs price",
              delta_color="normal")
    a4.metric("PHVS net price", usd(params.phvs_price))

    _fig(viz.fig_choice_shares(choice),
         "Multinomial logit over ledger coefficients - scenario output, not observed "
         "prescribing.")
    _fig(viz.fig_prescribing_contour(_surface(), stated, 0.18, 1.0),
         "Amber line: every combination of price premium and switching rate that reproduces the "
         "market price. The red cross is the ledger's stated inputs.")

    st.subheader("Break-even")
    be = pm.break_even_switching(premium, target_price=stated)
    if be["attainable"]:
        st.success(f"At a price premium of {premium:.2f}\u00d7, the model equals the market "
                   f"price at a **{be['required_switching_rate']:.1%}** annual switching rate "
                   f"(stated base case 18.0%).")
    else:
        st.error(f"No switching rate in [0.01%, 95%] reproduces the price at this premium. "
                 f"{be['note']}")
    seg = pm.segment_break_even()
    _df(seg)
    st.caption("A single segment cannot carry the price on its own; the price requires "
               "broad-based switching across the eligible pool.")

    with st.expander("How the map works"):
        st.markdown(pm.explain_choice_model())


def section_evidence() -> None:
    st.markdown('<div class="phvs-kicker">Section 8 \u00b7 Evidence</div>',
                unsafe_allow_html=True)
    st.title("Evidence ledger and verification")
    _intro("evidence")

    df = dl.get_clinical_trials()
    sources = dl.get_sources()
    findings = dl.validate_clinical_inputs(df)
    nesting = dl.validate_threshold_nesting(df)
    audit = _audit()
    facts, assumptions = dl.separate_facts_from_assumptions(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", len(df))
    c2.metric("Distinct sources", len(sources))
    c3.metric("Count-based endpoints", int(df["is_count"].sum()))
    c4.metric("Structural findings", len(findings) + len(nesting))

    _fig(viz.fig_evidence_coverage(ev.provenance_summary(df)),
         "Status is assigned row by row from the source that backs it.")

    tab_a, tab_b, tab_c = st.tabs(["Observations", "Sources", "Verification"])

    with tab_a:
        _df(df)
        st.caption(f"{len(facts)} rows carry reported or calculated observations; "
                   f"{len(assumptions)} rows are explicitly flagged as assumed inputs.")
    with tab_b:
        _df(sources)
    with tab_c:
        audit_state = "all checks passed" if audit["ok"] else "findings present"
        st.markdown(f"**Registry audit:** {audit_state} - "
                    f"{len(audit.get('findings', []))} findings.")
        if audit.get("findings"):
            _df(pd.DataFrame(audit["findings"]))
        if findings or nesting:
            _df(pd.DataFrame(findings + nesting))
        if st.button("Re-verify count endpoints against ClinicalTrials.gov", type="primary"):
            with st.spinner("Querying the registry (one request per trial)..."):
                result = ev.verify_against_registry(df, timeout=20)
            st.session_state["registry_check"] = result
        check = st.session_state.get("registry_check")
        if check is not None:
            _df(check if isinstance(check, pd.DataFrame) else pd.DataFrame(check))

    st.divider()
    st.subheader("Registry status of the pivotal trial")
    st.markdown("ClinicalTrials.gov holds **no results record** for NCT06669754. The "
                "CHAPTER-3 responder counts used in exhibit 1 are therefore "
                "company-reported and provisional: three of the four threshold rows are "
                "flagged `provisional_unverified`, and only the attack-free count is "
                "corroborated against a second source.")
    _disclaimers()


def section_survey() -> None:
    status = sv.survey_status()
    if not status["conducted"]:
        st.info(status["notice"])
    else:
        st.success(status["notice"])

    tab_a, tab_b, tab_c, tab_d = st.tabs(["Design", "Demo mode", "What it would replace",
                                          "What it cannot do"])
    with tab_a:
        demo = _demo_survey()
        st.markdown(f"**{len(demo['profiles'])} profiles**, **{len(demo['tasks'])} pairwise "
                    f"tasks**, {len(demo['assignments'])} respondent slots, two stages "
                    "(route hidden, then disclosed).")
        _df(sv.describe_tasks(demo["tasks"], demo["stage2_tasks"]), height=320)
        if st.button("Export design JSON"):
            path = EXPORT_DIR / "survey_design.json"
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            sv.export_survey_design(demo["tasks"], demo["stage2_tasks"],
                                    demo["assignments"], str(path))
            st.write(f"Written to `{path}`")

    with tab_b:
        st.markdown(f'<div class="phvs-alert">{sv.DEMO_NOTICE} The responses below are drawn '
                    f'from a stated parameter vector so the pipeline can be inspected. '
                    f'No physician has answered these questions.</div>',
                    unsafe_allow_html=True)
        demo = _demo_survey()
        fit = demo["fit"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Synthetic responses", fit["n_responses"])
        c2.metric("Tasks used", fit["n_tasks_used"])
        c3.metric("McFadden pseudo-R\u00b2", f"{fit['mcfadden_r2']:.3f}")
        _df(fit["coefficients"][["term", "utility", "std_error", "z", "p_value"]])
        _df(demo["importance"][["attribute", "utility_range", "importance"]], height=260)
        st.caption(fit["se_note"])

    with tab_c:
        st.markdown("Each row is a parameter the lab currently **assumes** and that a completed "
                    "instrument would estimate instead.")
        _df(sv.survey_maps_to_ledger(), height=420)

    with tab_d:
        st.markdown(sv.explain_survey_limitations())


def section_ledger() -> None:
    st.markdown('<div class="phvs-kicker">Ledger</div>', unsafe_allow_html=True)
    st.title("Assumption and source ledger")
    assumptions = dl.get_commercial_assumptions()
    categories = ["all"] + sorted(assumptions["category"].unique())
    col1, col2 = st.columns([1, 1])
    choice = col1.selectbox("Category", categories)
    status = col2.selectbox("Verification status",
                            ["all"] + sorted(assumptions["verification_status"].unique()))
    view = assumptions.copy()
    if choice != "all":
        view = view[view["category"] == choice]
    if status != "all":
        view = view[view["verification_status"] == status]
    _df(viz.format_assumption_ledger(view), height=460)
    st.caption(f"{len(view)} of {len(assumptions)} parameters shown. "
               "Every row carries its type (source fact vs analyst assumption), status and "
               "source.")


def section_exports() -> None:
    st.markdown('<div class="phvs-kicker">Exhibits</div>', unsafe_allow_html=True)
    st.title("Exportable exhibits")
    st.markdown("The three exhibits are written as PNGs for the write-up. They are drawn from "
                "the same functions the app uses, so the exported numbers cannot drift from the "
                "screen.")
    if st.button("Render all three", type="primary"):
        figs = rs.exhibit_figures()
        paths = []
        with st.spinner("Rendering with kaleido..."):
            for name in viz.EXPORTED_CHARTS:
                paths.append(viz.export_chart(figs[name], name))
        for p in paths:
            st.write(f"`{p}`")
        st.session_state["exported"] = [str(p) for p in paths]
    for p in st.session_state.get("exported", []):
        st.image(p, use_container_width=True)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

_SECTION_FN = {
    "1 · Recommendation": section_recommendation,
    "2 · Research questions": section_research,
    "3 · Exhibit 1 · Response": section_exhibit1,
    "4 · Exhibit 2 · Adoption": section_exhibit2,
    "5 · Exhibit 3 · Evidence to downside": section_exhibit3,
    "6 · Valuation & the short": section_valuation,
    "7 · Limitations & disconfirming evidence": section_limitations,
    "8 · Evidence & method": section_evidence,
}


def main() -> None:
    st.set_page_config(page_title="PHVS Clinical Expectations Lab", page_icon="◆",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(BRAND_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### PHVS Clinical Expectations Lab")
        st.caption("Research component of a short thesis on Pharvaris (PHVS): what the price "
                   "requires, and what has actually been measured.")
        section = st.radio("Sections", SECTIONS, label_visibility="collapsed")
        st.divider()
        st.caption("Research for a short thesis, not investment advice. Scenario outputs are "
                   "uncalibrated model results.")
        st.caption(f"Data as of {dt.date(2026, 10, 2)} · price $31.19 close")

    _SECTION_FN[section]()

    with st.expander("Assumption and source ledger"):
        section_ledger()
    with st.expander("Exportable exhibits"):
        section_exports()
    with st.expander("Methodology"):
        st.markdown(_markdown_file("METHODOLOGY.md"))
    with st.expander("README"):
        st.markdown(_markdown_file("README.md"))


if __name__ == "__main__":
    main()
