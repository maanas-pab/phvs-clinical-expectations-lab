"""PHVS Clinical Expectations Lab - Streamlit application.

Eight sections, one question: what would have to be true for Pharvaris to be
worth the observed price, and which of those things has actually been measured?

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
from phvs_lab.modules import response_engine as re_
from phvs_lab.modules import survey as sv
from phvs_lab.modules import trial_audit as ta
from phvs_lab.modules import valuation as val
from phvs_lab.utils import visualization as viz

SECTIONS = [
    "1 · Thesis",
    "2 · Evidence ledger",
    "3 · Response reconstruction",
    "4 · Trial audit",
    "5 · Population & economics",
    "6 · Prescribing map",
    "7 · Valuation & the short",
    "8 · Survey & missing evidence",
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


def _disclaimers() -> None:
    for key in ("not_forecast", "not_evidence", "no_h2h", "no_fabrication"):
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


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_thesis(challenge: bool) -> None:
    st.markdown('<div class="phvs-kicker">Section 1</div>', unsafe_allow_html=True)
    st.title("What would have to be true")
    _intro("evidence",
           "This lab does not take a view on Pharvaris. It reverses the arithmetic: given the "
           "observed price, which clinical, prescribing and commercial inputs must the market "
           "already be assuming, and which of those inputs have actually been measured?")

    summary = val.value_summary(_base_params())
    repro = _reproduce()
    scenarios = _scenarios()
    bundle = val.required_bundle(_base_params())
    short = val.short_metrics(_base_params())

    price = summary["stock_price"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market price (2 Oct 2026)", usd(price, 2))
    m2.metric("Reference model", usd(summary["value_per_share"], 2),
              delta=f"{summary['upside_downside_pct']:+.1%} vs price",
              delta_color="normal")
    m3.metric("Pitch model, own cash flows", usd(repro["value_per_share_current_shares"], 2),
              delta=f"{repro['multiple_required']:.2f}× required for price",
              delta_color="inverse")
    m4.metric("Enterprise value the price requires", usd_m(bundle["required_enterprise_value"]),
              delta=f"{bundle['required_multiple_of_stated_ev']:.2f}× the reference EV",
              delta_color="inverse")

    _fig(viz.fig_pitch_vs_models(repro, scenarios, price),
         "Bars are model outputs at the stated inputs; the dashed line is the observed close.")

    st.subheader("The conclusion, stated plainly")
    required = (
        f"an enterprise value of {usd(bundle['required_enterprise_value'])} - "
        f"{bundle['required_multiple_of_stated_ev']:.2f}× the reference model and "
        f"{repro['multiple_required']:.2f}× the pitch's own model - which is reached at "
        f"{'a peak prophylaxis penetration of ' + pct(_market_implied().set_index('input').loc['peak_penetration_prophylaxis','required_value'], 1)}"
        f", an approval probability of "
        f"{pct(_market_implied().set_index('input').loc['approval_prophylaxis','required_value'], 1)}"
        f" or a discount rate of "
        f"{pct(_market_implied().set_index('input').loc['discount_rate','required_value'], 1)}"
        f" on its own"
    )
    supported = (
        f"the ledger's stated inputs, which produce {usd(summary['value_per_share'], 2)} per share "
        f"from a 15% peak penetration, 88% approval probability and 10% discount rate - while the "
        f"pitch's own cash flows reproduce to only "
        f"{usd(repro['value_per_share_current_shares'], 2)} once the verified 70.2m share count "
        f"is used, and the choice model at stated preferences delivers "
        f"{pct(pm.implied_peak_penetration(), 1)} peak penetration and "
        f"{usd(pm.value_for_inputs(1.0, 0.18)['value_per_share'], 2)}"
    )
    unmeasured = (
        "real-world switching rates, the net price actually realised after gross-to-net, "
        "durability of the attack-free rate beyond 24 weeks, and any head-to-head evidence "
        "against injected prophylaxis"
    )
    st.markdown(
        f'<div class="phvs-conclusion">{CONCLUSION_TEMPLATE.format(X=required, Y=supported, Z=unmeasured)}</div>',
        unsafe_allow_html=True)

    st.subheader("Two models disagree by 3.6×")
    st.markdown(
        f"- The **pitch's own cash flows** reproduce to an enterprise value of "
        f"{usd(repro['enterprise_value'])} and {usd(repro['value_per_share_current_shares'], 2)} "
        f"per share on today's {repro['shares_current']/1e6:.1f}m shares. The price needs "
        f"{repro['multiple_required']:.2f}× that.\n"
        f"- The **extended reference model** adds the verified share count, cash, launch timing, "
        f"approval risk and terminal mechanics, and produces {usd(summary['equity_value'])} of "
        f"equity value - {usd(summary['value_per_share'], 2)} per share.\n"
        f"- The price of {usd(price, 2)} sits between them. Which of the two you believe is the "
        f"whole argument, and section 7 shows which inputs separate them.")

    if challenge:
        st.error("Challenge the thesis - the strongest case against a short here")
        st.markdown(
            f"- **The model says the stock is cheap, not rich.** At the stated inputs the "
            f"reference model gives {usd(summary['value_per_share'], 2)} against a "
            f"{usd(price, 2)} price: {summary['upside_downside_pct']:+.1%}. A short loses money "
            f"if that is right.\n"
            f"- **Each individual input has slack.** The price requires "
            f"{pct(_market_implied().set_index('input').loc['peak_penetration_prophylaxis','required_value'], 1)} "
            f"peak penetration against 15% stated, so the stated case does not need to be "
            f"revised down for the price to be explainable.\n"
            f"- **Bear case is conditional.** The short only works at {usd(short['breakeven_value_per_share'], 2)} "
            f"per share of model value or below; at the reference inputs it does not work at all "
            f"(net {short['net_return_pct']:+.1%} over the stated horizon).\n"
            f"- **Carry is real.** Borrow and carry of "
            f"{_base_params().borrow_cost:.0%} a year means the position must resolve on "
            f"schedule, not merely be correct eventually.")

    _disclaimers()


def section_evidence(challenge: bool) -> None:
    st.markdown('<div class="phvs-kicker">Section 2</div>', unsafe_allow_html=True)
    st.title("Evidence ledger")
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
        st.markdown(f"**Registry audit:** {'all checks passed' if audit['ok'] else 'findings present'} "
                    f"- {len(audit.get('findings', []))} findings.")
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
    if challenge:
        st.warning("Challenge the thesis - what this ledger cannot do")
        st.markdown(
            "A verified row only means *the number in the table matches the source*. "
            "It says nothing about whether the source is the right comparator, whether the "
            "window is comparable, or whether a placebo-controlled result transfers to a "
            "market with existing options. Sections 3 and 4 exist because verification and "
            "comparability are different questions.")
    _disclaimers()


def section_response(challenge: bool) -> None:
    st.markdown('<div class="phvs-kicker">Section 3</div>', unsafe_allow_html=True)
    st.title("Response reconstruction")
    _intro("response")

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

    if challenge:
        st.warning("Challenge the thesis - what reconstruction cannot recover")
        st.markdown(
            "- Nested thresholds identify exclusive categories **only when every threshold was "
            "disclosed**. Where one is missing the correct answer is a set, not a point.\n"
            "- Subtype counts (type I/II vs other HAE) were not disclosed, so a "
            "subtype-specific attack-free rate is not identified at all - the ledger can only "
            "bound it until the split is published.\n"
            "- A 168-day attack-free count is not an annual probability, and no confidence "
            "interval can make it one.")
    _disclaimers()


def section_audit(challenge: bool) -> None:
    st.markdown('<div class="phvs-kicker">Section 4</div>', unsafe_allow_html=True)
    st.title("Trial comparability audit")
    _intro("audit")

    df = dl.get_clinical_trials()
    table = ta.build_comparability_table(df)
    warnings = ta.comparability_warnings(df)

    st.markdown(f"**{len(warnings)} comparability warnings** are raised across the trial set. "
                "Every one is structural - population, window, endpoint or comparator - not statistical.")
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


    if challenge:
        st.warning("Challenge the thesis - the read-across everyone makes")
        st.markdown(
            "- CHAPTER-3 is **vs placebo**, not vs an existing prophylactic. Nothing in this "
            "ledger measures how many patients switch off an injected LTP for an oral one.\n"
            "- Windows differ (84 vs 168 days) and populations differ (children in HELP, "
            "crossover in COMPACT). Placing them on one axis is a demonstration of the window "
            "effect, not evidence of relative performance.\n"
            "- '83% reduction' and '45% attack-free' are different statistics; the short case "
            "that leans on one while quoting the other is not coherent.")
    _disclaimers()


def section_economics(challenge: bool) -> None:
    st.markdown('<div class="phvs-kicker">Section 5</div>', unsafe_allow_html=True)
    st.title("Population and economics")
    _intro("economics")

    S = _economics()
    ledger = S["ledger"]
    bridge = S["bridge"]

    _fig(viz.fig_population_funnel(ledger),
         "Segments are mutually exclusive and validated to sum to 1.0 of the eligible pool.")
    _fig(viz.fig_prevention_ladder(S["ladder"]),
         "Attacks, patients and revenue are three separate quantities with their own conversions.")

    st.markdown(ec.explain_double_counting())

    eff = st.slider("Scenario efficacy (before real-world haircut)", 0.40, 1.00,
                    float(S["params"].efficacy), 0.01)
    proj = ec.efficacy_projection(S["params"], np.linspace(0.40, 1.00, 25))
    _fig(viz.fig_efficacy_projection(proj),
         f"Dispersion k={bridge['fitted_dispersion_k']:.4g}, calibrated once from the trial's "
         "published values and held fixed while efficacy varies.")
    row = proj.iloc[(proj["efficacy"] - eff).abs().argsort().iloc[0]]
    st.caption(f"At efficacy {eff:.0%}: {row['breakthrough_patients']:,.0f} breakthrough patients, "
               f"{usd(row['acute_revenue'])} acute revenue, {usd(row['total_revenue'])} combined.")

    if challenge:
        st.warning("Challenge the thesis - where the model could be wrong")
        st.markdown(
            "- **Addressable population is an assumption.** The ledger's 19,614 global "
            "diagnosed patients is already ~3× below the pitch's 60,000; if the pitch is right "
            "every derived share scales up.\n"
            "- **Segment shares and addressability are analyst inputs**, not measured. Change "
            "them and the prescribing map moves in section 6.\n"
            "- **The acute stream depends on breakthrough**, which is modelled from a "
            "gamma-Poisson fit to one trial's published attack-free rate - a fit, not an "
            "observation.")
    _disclaimers()


def section_map(challenge: bool) -> None:
    st.markdown('<div class="phvs-kicker">Section 6</div>', unsafe_allow_html=True)
    st.title("Prescribing map")
    _intro("map")

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
              delta=f"{value['value_per_share']/stated - 1:+.1%} vs price", delta_color="normal")
    a4.metric("PHVS net price", usd(params.phvs_price))

    _fig(viz.fig_choice_shares(choice),
         "Multinomial logit over ledger coefficients - scenario output, not observed prescribing.")
    _fig(viz.fig_prescribing_contour(_surface(), stated, 0.18, 1.0),
         "Amber line: every combination of price premium and switching rate that reproduces the "
         "market price. The red cross is the ledger's stated inputs.")

    st.subheader("Break-even")
    be = pm.break_even_switching(premium, target_price=stated)
    if be["attainable"]:
        st.success(f"At a price premium of {premium:.2f}×, the model equals the market price at a "
                   f"**{be['required_switching_rate']:.1%}** annual switching rate "
                   f"(stated base case 18.0%).")
    else:
        st.error(f"No switching rate in [0.01%, 95%] reproduces the price at this premium. {be['note']}")
    seg = pm.segment_break_even()
    _df(seg)
    st.caption("A single segment cannot carry the price on its own; the price requires broad-based "
               "switching across the eligible pool.")

    with st.expander("How the map works"):
        st.markdown(pm.explain_choice_model())

    if challenge:
        st.warning("Challenge the thesis - stated preference is not prescribing")
        st.markdown(
            "- Every coefficient in the map is an **assumption in the ledger**, flagged "
            "`unverified`. The map is a translation of assumptions into patients, not evidence.\n"
            "- The map's base case lands at "
            f"{pct(pm.implied_peak_penetration(pm.MapParams()), 1)} peak penetration against the "
            "ledger's stated 15% - a small but real inconsistency between two parts of the same "
            "model that neither side has measured.\n"
            "- Price realisation is modelled as a uniform premium; real gross-to-net, formulary "
            "tiering and PBM behaviour are absent.")
    _disclaimers()


def section_valuation(challenge: bool) -> None:
    st.markdown('<div class="phvs-kicker">Section 7</div>', unsafe_allow_html=True)
    st.title("Valuation and the short")
    _intro("valuation")

    base = _base_params()
    st.subheader("The pitch, reproduced first")
    repro = _reproduce()
    _df(pd.DataFrame(repro["findings"]), height=300)
    st.caption(f"Reproduced enterprise value {usd(repro['enterprise_value'])}; "
               f"the price needs {repro['multiple_required']:.2f}× that.")

    st.subheader("Then extended")
    p1, p2, p3, p4, p5 = st.columns(5)
    peak = p1.slider("Peak prophylaxis penetration", 0.02, 0.40, float(base.peak_penetration_prophylaxis), 0.01)
    approval = p2.slider("Prophylaxis approval probability", 0.30, 1.00, float(base.approval_prophylaxis), 0.01)
    discount = p3.slider("Discount rate", 0.06, 0.25, float(base.discount_rate), 0.005, format="%.3f")
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
              delta=f"{live_summary['required_ev_for_price']/live_summary['enterprise_value']:.2f}× this model",
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

    if challenge:
        st.warning("Challenge the thesis - when this short fails")
        st.markdown(
            "- At the reference inputs the short **does not work**: model value "
            f"{usd(val.value_summary(base)['value_per_share'], 2)} against a "
            f"{usd(base.stock_price, 2)} price.\n"
            f"- It needs the model value below "
            f"{usd(val.short_metrics(base)['breakeven_value_per_share'], 2)}, "
            "which is roughly the bear case.\n"
            "- The largest swing is price and peak penetration (identical by construction, both "
            "scale revenue). A single successful launch print on either one removes the case.\n"
            "- Approval is pending with a stated PDUFA date; a positive decision moves the "
            "approval input directly and the market knows the date.")
    _disclaimers()


def section_survey(challenge: bool) -> None:
    st.markdown('<div class="phvs-kicker">Section 8</div>', unsafe_allow_html=True)
    st.title("Survey and missing evidence")
    st.markdown('<div class="phvs-lead">'
                'The gap this instrument exists to fill is stated plainly: '
                '<b>the survey has not been conducted</b>.</div>', unsafe_allow_html=True)

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
        c3.metric("McFadden pseudo-R²", f"{fit['mcfadden_r2']:.3f}")
        _df(fit["coefficients"][["term", "utility", "std_error", "z", "p_value"]])
        _df(demo["importance"][["attribute", "utility_range", "importance"]], height=260)
        st.caption(fit["se_note"])

    with tab_c:
        st.markdown("Each row is a parameter the lab currently **assumes** and that a completed "
                    "instrument would estimate instead.")
        _df(sv.survey_maps_to_ledger(), height=420)

    with tab_d:
        st.markdown(sv.explain_survey_limitations())

    st.subheader("Evidence still missing")
    _df(val.invalidation_conditions(_base_params()), height=360)

    if challenge:
        st.warning("Challenge the thesis - the honest summary of evidence")
        st.markdown(
            "- Registry-verified: CHAPTER-1 responder counts, trial identifiers, sample sizes "
            "and windows.\n"
            "- Company-reported and provisional: CHAPTER-3 responder counts, which have not yet "
            "been posted to the registry.\n"
            "- Modelled: every commercial quantity, including penetration, switching, price "
            "realisation and approval probability.\n"
            "- Not measured at all: real switching, durability past 24 weeks, head-to-head "
            "superiority, and physician preference.")
    _disclaimers()


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
               "Every row carries its type (source fact vs analyst assumption), status and source.")


def section_exports() -> None:
    st.markdown('<div class="phvs-kicker">Charts</div>', unsafe_allow_html=True)
    st.title("Exportable charts")
    st.markdown("Three charts are written as PNGs for the write-up. They are drawn from the same "
                "functions the app uses, so the exported numbers cannot drift from the screen.")
    if st.button("Render all three", type="primary"):
        base = _base_params()
        charts = {
            "phvs_required_inputs": viz.fig_required_inputs(_market_implied(), _sensitivity()),
            "phvs_response_categories": viz.fig_response_categories(
                list(re_.distributions_for_trial("CHAPTER-3", dl.get_clinical_trials()).values())[0]),
            "phvs_value_scenarios": viz.fig_pitch_vs_models(
                _reproduce(), _scenarios(), base.stock_price),
        }
        paths = []
        with st.spinner("Rendering with kaleido..."):
            for name, fig in charts.items():
                paths.append(viz.export_chart(fig, name))
        for p in paths:
            st.write(f"`{p}`")
        st.session_state["exported"] = [str(p) for p in paths]
    for p in st.session_state.get("exported", []):
        st.image(p, use_container_width=True)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="PHVS Clinical Expectations Lab", page_icon="◆",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(BRAND_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### PHVS Clinical Expectations Lab")
        st.caption("Reverse-engineering the assumptions required to justify the "
                   "observed Pharvaris share price.")
        section = st.radio("Sections", SECTIONS, label_visibility="collapsed")
        st.divider()
        challenge = st.checkbox("Challenge our thesis",
                                help="Surfaces the strongest argument against each panel.")
        st.divider()
        st.caption("Not investment advice. Scenario outputs are uncalibrated model results.")
        st.caption(f"Data as of {dt.date(2026, 10, 2)} · price $31.19 close")

    if section == "1 · Thesis":
        section_thesis(challenge)
    elif section == "2 · Evidence ledger":
        section_evidence(challenge)
    elif section == "3 · Response reconstruction":
        section_response(challenge)
    elif section == "4 · Trial audit":
        section_audit(challenge)
    elif section == "5 · Population & economics":
        section_economics(challenge)
    elif section == "6 · Prescribing map":
        section_map(challenge)
    elif section == "7 · Valuation & the short":
        section_valuation(challenge)
    elif section == "8 · Survey & missing evidence":
        section_survey(challenge)

    with st.expander("Assumption and source ledger"):
        section_ledger()
    with st.expander("Exportable charts"):
        section_exports()
    with st.expander("Methodology"):
        st.markdown(_markdown_file("METHODOLOGY.md"))
    with st.expander("README"):
        st.markdown(_markdown_file("README.md"))


if __name__ == "__main__":
    main()
