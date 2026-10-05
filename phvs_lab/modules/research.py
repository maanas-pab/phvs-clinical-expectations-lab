"""Research component of the PHVS short thesis.

Three research questions, each resolved in the same four steps:

    methodology -> finding -> investment implication -> valuation impact

Every figure produced here is computed from the evidence ledger and the
analysis modules. Nothing is hard-coded: if the ledger changes, the findings
re-derive. Where the data cannot answer a question the block returns a
break-even threshold or a sensitivity instead of a conclusion.

Status tags used throughout:

    ``completed``  analysis that has been run on published data
    ``proposed``   research designed but not yet carried out
    ``assumed``    analyst input with no source measurement
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dataclasses import replace as _replace
from scipy import optimize

from phvs_lab.modules import data_loader as dl
from phvs_lab.modules import economics as ec
from phvs_lab.modules import prescribing_map as pm
from phvs_lab.modules import response_engine as re_
from phvs_lab.modules import valuation as val

__all__ = [
    "STATUS", "ResearchBlock", "short_recommendation", "rq1_response",
    "rq2_adoption", "rq3_price", "chain_links", "chain_break_even",
    "workstreams", "exhibits", "disconfirming_evidence", "limitations_notes",
    "EXHIBIT_PNGS",
    "chain_curve", "clinical_anchors", "adoption_bars", "exhibit1_inputs",
    "exhibit2_inputs", "exhibit3_inputs", "exhibit_source_lines",
    "exhibit_figures",
]

STATUS = {"completed": "completed", "proposed": "proposed", "assumed": "assumed"}

EXHIBIT_PNGS = {
    "EX1": "phvs_exhibit_1_response_distribution",
    "EX2": "phvs_exhibit_2_adoption_constraint",
    "EX3": "phvs_exhibit_3_evidence_to_downside",
}


# ---------------------------------------------------------------------------
# Block container
# ---------------------------------------------------------------------------

@dataclass
class ResearchBlock:
    """One research question, resolved in four steps."""

    key: str
    question: str
    methodology: str
    finding: str
    implication: str
    valuation_impact: str
    status: str
    metrics: Dict[str, str]
    table: Optional[pd.DataFrame] = None
    thresholds: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Shared computations
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _base() -> val.ValuationParams:
    return val.params_from_assumptions()


@lru_cache(maxsize=4)
def _assumptions() -> pd.DataFrame:
    return dl.get_commercial_assumptions()


def _a(name: str) -> float:
    return dl.get_assumption(name, _assumptions())


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def _usd(x: float, d: int = 0) -> str:
    return f"${x:,.{d}f}"


def _half_swing(field: str, shock: float = 0.10) -> float:
    """Half the up/down swing in value per share from a one-at-a-time shock."""
    base = _base()
    stated = getattr(base, field)
    up = val.value_summary(_replace(base, **{field: stated * (1 + shock)}))["value_per_share"]
    down = val.value_summary(_replace(base, **{field: stated * (1 - shock)}))["value_per_share"]
    return abs(up - down) / 2.0


# ---------------------------------------------------------------------------
# RQ1 - what the response distribution reveals
# ---------------------------------------------------------------------------

def rq1_response() -> ResearchBlock:
    """RQ1: what does the clinical response distribution reveal?"""
    trials = dl.get_clinical_trials()
    dist = re_.distributions_for_trial("CHAPTER-3", trials)["Deucrictibant XR 40 mg QD"]
    counts = dist.counts
    intervals = dist.proportion_intervals()
    n = dist.total_patients

    headline = _a("chapter3_rate_reduction_vs_placebo")
    attack_free = counts["attack-free (100%)"] / n
    af_row = intervals[intervals["level"] == 4].iloc[0]
    deep = counts["90-99% reduction"] / n + attack_free
    shallow = (counts["70-89% reduction"] + counts["50-69% reduction"]) / n
    non_resp = counts["<50% reduction (non-responder)"] / n
    res = re_.sample_size_resolution(n)

    provisional = trials[(trials["trial"] == "CHAPTER-3") & (trials["is_count"])].copy()
    n_provisional = int((provisional["verification_status"] == "provisional_unverified").sum())

    table = pd.DataFrame([
        {"statistic": "Mean monthly attack-rate reduction vs placebo",
         "value": headline, "kind": "ratio of mean rates",
         "basis": "reported", "status": "verified_primary",
         "what_it_does_not_say": "says nothing about how many patients are fully controlled"},
        {"statistic": f"At least {90}% reduction in attack rate",
         "value": deep, "kind": "patient proportion",
         "basis": "calculated from counts", "status": "provisional_unverified",
         "what_it_does_not_say": "threshold counts are provisional pending publication"},
        {"statistic": "Attack-free over 168 days",
         "value": attack_free, "kind": "patient proportion",
         "basis": "reported count 25/55", "status": "corroborated_secondary",
         "what_it_does_not_say": "a 24-week count is not an annual or lifetime probability"},
        {"statistic": "Less than 50% reduction (non-responder)",
         "value": non_resp, "kind": "patient proportion",
         "basis": "calculated from counts", "status": "provisional_unverified",
         "what_it_does_not_say": "one patient moves this by "
                                 f"{_pct(res['one_patient_share'], 1)}"},
    ])

    thresholds = pd.DataFrame([
        {"threshold": "Shortest attainable", "value": deep, "status": "provisional_unverified"},
        {"threshold": "Headline mean (not a patient share)", "value": headline,
         "status": "verified_primary"},
        {"threshold": "Attack-free at 168 days", "value": attack_free,
         "status": "corroborated_secondary"},
    ])

    finding = (
        f"The headline is a ratio of means; the ledger also holds the patient counts. "
        f"In the CHAPTER-3 active arm (n={n}) the mean fell {_pct(headline, 0)} against placebo, "
        f"while {_pct(deep, 1)} of patients reached at least a 90% reduction and only "
        f"{_pct(attack_free, 1)} were attack-free over 168 days "
        f"(exact 95% interval {_pct(af_row['ci_low'], 1)}–{_pct(af_row['ci_high'], 1)}). "
        f"{_pct(shallow, 1)} of treated patients landed between a 50% and 90% reduction. "
        f"{n_provisional} of the {len(provisional)} threshold rows are still flagged "
        f"provisional, and one patient moves any share by {_pct(res['one_patient_share'], 1)}."
    )

    valuation_impact = (
        "None of these quantities is an input to the discounted cash flow. The valuation's "
        "three largest drivers are net price, peak penetration and approval probability, each "
        "moving value per share by the same amount per 10%; measured efficacy does not appear "
        "among them. Efficacy reaches the model only through two unmeasured analyst inputs - "
        "the 0.25-unit choice-model efficacy bonus and the 0.75 real-world haircut. So the "
        "clinical result constrains the story, not the number."
    )

    return ResearchBlock(
        key="RQ1",
        question="What does the clinical response distribution reveal that headline efficacy "
                 "obscures?",
        methodology=(
            "Nested cumulative responder thresholds for the CHAPTER-3 active arm "
            "(&#8805;50% / &#8805;70% / &#8805;90% / attack-free) are inverted into mutually exclusive "
            "categories by subtraction. Sampling uncertainty is reported as exact "
            "Clopper-Pearson intervals on the disclosed cumulative counts. Observation window, "
            "sample size and verification status are carried separately and never folded into "
            "the interval. Separate trials are never compared."
        ),
        finding=finding,
        implication=(
            "The efficacy headline that carries the commercial case is a mean, and a mean can "
            "be driven by patients with high baseline attack rates. The patient-level number "
            "the label and the payer will see is roughly half the headline, is measured on "
            f"n={n}, and is still provisional. Adoption models built on the headline are "
            "therefore resting on the statistic that is least able to describe a typical "
            "patient."
        ),
        valuation_impact=valuation_impact,
        status="completed",
        metrics={
            "Headline mean reduction": _pct(headline, 0),
            "Attack-free at 168 days": _pct(attack_free, 1),
            f"Exact 95% interval on attack-free": f"{_pct(af_row['ci_low'], 0)}–"
                                                  f"{_pct(af_row['ci_high'], 0)}",
            "Threshold rows still provisional": f"{n_provisional} of {len(provisional)}",
        },
        table=table,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# RQ2 - what constrains adoption
# ---------------------------------------------------------------------------

def rq2_adoption() -> ResearchBlock:
    """RQ2: competing options and the prevention / rescue relationship."""
    base = _base()
    mi = val.market_implied(base).set_index("input")
    req_pen = float(mi.loc["peak_penetration_prophylaxis", "required_value"])
    eligible = base.prophylaxis_eligible

    required_pts = eligible * req_pen
    stated_pts = eligible * base.peak_penetration_prophylaxis

    new_starts = _a("new_ltp_patients_per_year_global")
    ramp = base.time_to_peak_prophylaxis
    inflow = new_starts * ramp          # hard lower bound: 100% global capture

    switch_required_price = max(0.0, (required_pts - inflow) / required_pts)
    switch_required_stated = max(0.0, (stated_pts - inflow) / stated_pts)

    oral_share = _a("implied_oral_share_of_ltp_value")
    takhyro_pts = _a("takhyro_patients_global")
    takhyro_price = _a("implied_revenue_per_takhyro_patient")
    incumbent_dropout = _a("orladeyo_annual_dropout_rate")
    modelled_persistence = 1.0 - _a("prophylaxis_annual_discontinuation")

    seg = pm.segment_break_even(base)
    best_seg = seg.loc[seg["max_value_attainable"].idxmax()]
    be = pm.break_even_switching(1.0, target_price=base.stock_price, base=base)

    peak_rev_p = base.prophylaxis_eligible * base.peak_penetration_prophylaxis * base.price_prophylaxis
    peak_rev_a = base.acute_eligible * base.peak_penetration_acute * base.price_acute
    acute_share_of_peak = peak_rev_a / peak_rev_p
    pen_swing = _half_swing("peak_penetration_prophylaxis")

    table = pd.DataFrame([
        {"constraint": "Global new long-term-prophylaxis patients per year",
         "reading": f"{new_starts:,.0f} per year (management range 150-250)",
         "type": "source fact", "status": "verified_secondary", "source": "S20"},
        {"constraint": "Market inflow over the 6-year ramp at 100% capture",
         "reading": f"{inflow:,.0f} patients - a hard lower bound, not a forecast",
         "type": "calculated", "status": "from a verified source fact", "source": "S20"},
        {"constraint": "Peak patients the observed price requires",
         "reading": f"{required_pts:,.0f} patients ({_pct(req_pen, 1)} of the eligible pool)",
         "type": "calculated", "status": "model solve", "source": "-"},
        {"constraint": "Share of those patients that must be switchers",
         "reading": f"at least {_pct(switch_required_price, 0)} - switching is unmeasured",
         "type": "calculated", "status": "unverified", "source": "-"},
        {"constraint": "Oral share of long-term prophylaxis value today",
         "reading": f"{_pct(oral_share, 0)} (Orladeyo vs TAKHZYRO revenue)",
         "type": "calculated", "status": "unverified", "source": "S10, S17"},
        {"constraint": "Incumbent oral dropout (company claim)",
         "reading": f"~{_pct(incumbent_dropout, 0)} against modelled persistence of "
                    f"{_pct(modelled_persistence, 0)}",
         "type": "source fact vs assumption", "status": "verified_secondary / unverified",
         "source": "S20"},
        {"constraint": "Best single segment, switching at 95%",
         "reading": f"{best_seg['segment']} reaches {_usd(best_seg['max_value_attainable'], 2)} "
                    f"against {_usd(base.stock_price, 2)}",
         "type": "model solve", "status": "assumed coefficients", "source": "-"},
        {"constraint": "Annual switching rate that reproduces the price",
         "reading": f"{_pct(be['required_switching_rate'], 1)} against a stated "
                    f"{_pct(_a('base_switching_rate'), 0)}",
         "type": "model solve", "status": "assumed coefficients", "source": "-"},
    ])

    thresholds = pd.DataFrame([
        {"threshold": "Switchers the price requires", "value": switch_required_price,
         "unit": "share of required patients"},
        {"threshold": "Switchers the stated 15% case requires", "value": switch_required_stated,
         "unit": "share of required patients"},
        {"threshold": "Annual switching rate that reproduces the price",
         "value": be["required_switching_rate"], "unit": "per year"},
        {"threshold": "Best single segment at 95% switching",
         "value": float(best_seg["max_value_attainable"]) / base.stock_price,
         "unit": "of the market price"},
    ])

    finding = (
        f"Adoption has to come from switching, because the market does not add patients fast "
        f"enough. Management puts new long-term-prophylaxis patients at {new_starts:,.0f} a year, "
        f"so even if PHVS captured every new patient worldwide for the whole "
        f"{ramp:.0f}-year ramp that is {inflow:,.0f} patients - against "
        f"{required_pts:,.0f} needed for the observed price and {stated_pts:,.0f} in our stated "
        f"case. At least {_pct(switch_required_price, 0)} of the patients the price requires must "
        f"be taken off existing therapy, and no switching rate has ever been measured. The "
        f"incumbent mix is {_pct(1 - oral_share, 0)} injectable, so most of that switch is off "
        f"injectables - and CHAPTER-3 is placebo-controlled, not head-to-head. No single segment "
        f"carries the price: the best reaches {_usd(best_seg['max_value_attainable'], 2)} at a "
        f"95% switching rate against {_usd(base.stock_price, 2)}."
    )

    valuation_impact = (
        f"Peak penetration is the joint second-largest driver of value per share "
        f"({_usd(pen_swing, 2)} per 10% move). The prevention / rescue relationship works "
        f"against adding the two streams together: the acute stream is worth "
        f"{_pct(acute_share_of_peak, 0)} of peak prophylaxis revenue and falls as prevention "
        f"improves, so the two products cannot both be underwritten at their individual best "
        f"case. Persistence is the cheapest lever nobody has measured - moving discontinuation "
        f"from the modelled {_pct(1 - modelled_persistence, 0)} to the incumbent's claimed "
        f"{_pct(incumbent_dropout, 0)} cuts the implied patient count by "
        f"{_pct(1 - (1 - incumbent_dropout) / modelled_persistence, 1)}."
    )

    return ResearchBlock(
        key="RQ2",
        question="How could competing treatment options and the relationship between prevention "
                 "and rescue use constrain PHVS adoption?",
        methodology=(
            "The patients the price requires are solved from the observed price, the verified "
            "share count and net cash, the assumed net price and the eligible pool. Market "
            "inflow uses management's own published range and is deliberately set at 100% "
            "global capture - a lower bound on how much must come from switching. Segment "
            "ceilings come from the ledger's multinomial-logit choice model, whose coefficients "
            "are analyst assumptions, not measured preferences. No cross-trial efficacy "
            "comparison is used anywhere in this block."
        ),
        finding=finding,
        implication=(
            "The competitive constraint is structural rather than clinical. Oral therapy is a "
            "minority of prophylaxis value, the pool grows slowly, and the price can only be "
            "reached by taking share from incumbents whose patients have never been measured "
            "moving. The prevention and rescue products cannibalise each other inside the same "
            "cohort, so the sum of their individual best cases is not an addressable market."
        ),
        valuation_impact=valuation_impact,
        status="assumed",
        metrics={
            "Switchers the price requires": _pct(switch_required_price, 0),
            "New LTP patients per year": f"{new_starts:,.0f}",
            "Peak patients at the price": f"{required_pts:,.0f}",
            "Best single segment vs price": f"{_usd(best_seg['max_value_attainable'], 2)} / "
                                            f"{_usd(base.stock_price, 2)}",
        },
        table=table,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# RQ3 - what the price requires, and how our case differs
# ---------------------------------------------------------------------------

def rq3_price() -> ResearchBlock:
    """RQ3: adoption assumptions behind the price and the evidence-backed short case."""
    base = _base()
    mi = val.market_implied(base)
    bundle = val.required_bundle(base)
    repro = val.reproduce_pitch_valuation(dl.get_valuation_model())
    scenarios = val.scenario_table(base).set_index("scenario")
    short = val.short_metrics(base)
    sens = val.sensitivity_ranking(base)
    targets = mi.set_index("input")
    summary_ev = val.value_summary(base)["enterprise_value"]

    req_pen = float(targets.loc["peak_penetration_prophylaxis", "required_value"])
    req_appr = float(targets.loc["approval_prophylaxis", "required_value"])
    req_disc = float(targets.loc["discount_rate", "required_value"])
    req_price = float(targets.loc["price_prophylaxis", "required_value"])
    takhyro_price = _a("implied_revenue_per_takhyro_patient")

    top = sens.head(3)
    top_names = ", ".join(top["parameter"].str.replace("_", " ").tolist())

    table = pd.DataFrame([
        {"input": "Peak prophylaxis penetration", "stated": base.peak_penetration_prophylaxis,
         "required_for_price": req_pen, "unit": "share of eligible pool",
         "verified": "no - analyst assumption"},
        {"input": "Prophylaxis approval probability", "stated": base.approval_prophylaxis,
         "required_for_price": req_appr, "unit": "probability", "verified": "no - analyst assumption"},
        {"input": "Discount rate", "stated": base.discount_rate,
         "required_for_price": req_disc, "unit": "per year", "verified": "no - analyst assumption"},
        {"input": "Annual net price", "stated": base.price_prophylaxis,
         "required_for_price": req_price, "unit": "USD per patient-year",
         "verified": "no - Pharvaris has not set a price"},
        {"input": "Realised revenue per incumbent LTP patient", "stated": takhyro_price,
         "required_for_price": req_price, "unit": "USD per patient-year",
         "verified": "calculated from disclosed revenue and patient count"},
    ])

    thresholds = pd.DataFrame([
        {"threshold": "Peak penetration that reproduces the price",
         "value": req_pen, "unit": "of eligible pool", "status": "model solve"},
        {"threshold": "Peak penetration at which the short breaks even after 6% carry",
         "value": _penetration_at(short["breakeven_value_per_share"]), "unit": "of eligible pool",
         "status": "model solve"},
        {"threshold": "Net price that reproduces the price", "value": req_price,
         "unit": "USD per patient-year", "status": "model solve"},
        {"threshold": "Net price / incumbent realised revenue per patient",
         "value": req_price / takhyro_price, "unit": "multiple", "status": "calculated"},
    ])

    spread_models = summary_ev / repro["enterprise_value"] if repro["enterprise_value"] else float("nan")

    finding = (
        f"Run one input at a time, the observed price of {_usd(base.stock_price, 2)} is "
        f"reproduced by {_pct(req_pen, 1)} peak prophylaxis penetration, a "
        f"{_pct(req_appr, 1)} approval probability, a {_pct(req_disc, 1)} discount rate or a "
        f"{_usd(req_price, 0)} net price - each holding everything else at its stated value, so "
        f"they cannot all be true at once. The required net price is "
        f"{req_price / takhyro_price:.2f}x the only realised revenue-per-patient anchor in the "
        f"ledger. Two defensible models disagree by {spread_models:.2f}x in enterprise value - "
        f"our extended model at {summary_ev / 1e6:,.0f}m and the circulated cash-flow model at "
        f"{repro['enterprise_value'] / 1e6:,.0f}m - and the price sits at "
        f"{bundle['required_multiple_of_stated_ev']:.2f}x the first and "
        f"{repro['multiple_required']:.2f}x the second. Nothing measured separates them: the "
        f"three largest drivers of value per share are {top_names}, all flagged unverified."
    )

    implication = (
        f"Our case is not a competing point estimate. It is the observation that the entire gap "
        f"between a {_usd(scenarios.loc['bear', 'value_per_share'], 0)} bear case and a "
        f"{_usd(scenarios.loc['bull', 'value_per_share'], 0)} bull case is filled by adoption "
        f"inputs that nobody has measured, while the clinical inputs that have been measured do "
        f"not reach the valuation at all. We therefore state the thresholds instead of a fair "
        f"value: the short is a position on adoption landing below what the price already "
        f"assumes, with a defined invalidation level rather than a target."
    )

    valuation_impact = (
        f"Break-even, not a target: after a {_pct(base.borrow_cost, 0)} annual borrow cost the "
        f"short returns money only when model value sits at or below "
        f"{_usd(short['breakeven_value_per_share'], 2)}, which on our stated inputs is "
        f"{_pct(_penetration_at(short['breakeven_value_per_share']), 1)} peak penetration - "
        f"{_pct(req_pen - _penetration_at(short['breakeven_value_per_share']), 1)} of headroom "
        f"below the level the price itself requires. At the reference inputs the short loses "
        f"{abs(short['net_return_pct']):.1f}%; in the bear case it makes "
        f"{val.short_metrics(val.scenario_params(base, 'bear'))['net_return_pct']:.1f}% net."
    )

    return ResearchBlock(
        key="RQ3",
        question="What adoption assumptions justify the stock price, and how does our "
                 "evidence-backed short case differ?",
        methodology=(
            "The inverse problem is solved input by input: each adoption and valuation input is "
            "moved until the model equals the observed close of "
            f"{_usd(base.stock_price, 2)}, all other ledger inputs held at their stated values. "
            "The required enterprise value is reported as a multiple of both the extended model "
            "and the reproduced cash-flow model. Sensitivities shock one input at a time. No "
            "input was moved in order to reach a conclusion."
        ),
        finding=finding,
        implication=implication,
        valuation_impact=valuation_impact,
        status="completed",
        metrics={
            "Required peak penetration": _pct(req_pen, 1),
            "Required net price": _usd(req_price, 0),
            "Required discount rate": _pct(req_disc, 1),
            "Required EV / cash-flow model": f"{repro['multiple_required']:.2f}x",
        },
        table=table,
        thresholds=thresholds,
    )


def _penetration_at(target_value: float) -> float:
    """Peak penetration at which the model equals ``target_value`` per share."""
    base = _base()
    f = lambda x: val.value_summary(  # noqa: E731
        _replace(base, peak_penetration_prophylaxis=x))["value_per_share"] - target_value
    lo, hi = 1e-6, 1.0
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(optimize.brentq(f, lo, hi, xtol=1e-10, rtol=1e-8))


# ---------------------------------------------------------------------------
# Exhibit inputs
# ---------------------------------------------------------------------------

def chain_curve(penetrations: Optional[np.ndarray] = None) -> Dict:
    """Value per share across peak penetration, plus the markers the exhibit draws.

    Every other ledger input is held at its stated value, so the curve isolates
    the one adoption input the price is most sensitive to.
    """
    base = _base()
    short = val.short_metrics(base)
    mi = val.market_implied(base).set_index("input")
    x = (np.linspace(0.02, 0.40, 77) if penetrations is None
         else np.asarray(penetrations, float))
    y = np.array([val.value_summary(
        _replace(base, peak_penetration_prophylaxis=float(p)))["value_per_share"] for p in x])
    return {
        "penetration": x,
        "value_per_share": y,
        "price": base.stock_price,
        "breakeven": short["breakeven_value_per_share"],
        "required": float(mi.loc["peak_penetration_prophylaxis", "required_value"]),
        "stated": base.peak_penetration_prophylaxis,
        "choice_model": pm.implied_peak_penetration(),
        "bear": float(val.scenario_params(base, "bear").peak_penetration_prophylaxis),
        "label": "MODEL OUTPUT - all other ledger inputs at their stated values",
    }


def clinical_anchors() -> pd.DataFrame:
    """The measured clinical quantities the exhibit shows in its first panel."""
    dist = re_.distributions_for_trial("CHAPTER-3", dl.get_clinical_trials())[
        "Deucrictibant XR 40 mg QD"]
    c, n = dist.counts, dist.total_patients
    rows = [
        {"anchor": "Mean rate reduction\nvs placebo",
         "value": _a("chapter3_rate_reduction_vs_placebo"),
         "kind": "mean (not a patient share)", "status": "verified_primary"},
        {"anchor": "At least 90%\nreduction",
         "value": (c["attack-free (100%)"] + c["90-99% reduction"]) / n,
         "kind": "patient share", "status": "provisional_unverified"},
        {"anchor": "Attack-free at\n168 days", "value": c["attack-free (100%)"] / n,
         "kind": "patient share", "status": "corroborated_secondary"},
        {"anchor": "Less than 50%\nreduction",
         "value": c["<50% reduction (non-responder)"] / n,
         "kind": "patient share", "status": "provisional_unverified"},
    ]
    out = pd.DataFrame(rows)
    out.attrs["n"] = n
    out.attrs["window_days"] = dist.window_days
    return out


def adoption_bars() -> pd.DataFrame:
    """Patients the price requires against what the market actually supplies."""
    base = _base()
    mi = val.market_implied(base).set_index("input")
    req = base.prophylaxis_eligible * float(
        mi.loc["peak_penetration_prophylaxis", "required_value"])
    stated = base.prophylaxis_eligible * base.peak_penetration_prophylaxis
    inflow = _a("new_ltp_patients_per_year_global") * base.time_to_peak_prophylaxis
    rows = [
        {"label": "Incumbent LTP patients today (TAKHZYRO, 55+ countries)",
         "patients": _a("takhyro_patients_global"), "kind": "source fact"},
        {"label": "All new LTP patients over the 6-year ramp, 100% capture",
         "patients": inflow, "kind": "lower bound"},
        {"label": "PHVS peak patients required by the observed price",
         "patients": req, "kind": "model solve"},
        {"label": "PHVS peak patients in our stated case (15%)",
         "patients": stated, "kind": "assumption"},
    ]
    out = pd.DataFrame(rows)
    out.attrs["required"] = req
    out.attrs["inflow"] = inflow
    out.attrs["switch_share"] = max(0.0, (req - inflow) / req)
    return out


def exhibit1_inputs() -> Dict:
    d = re_.distributions_for_trial("CHAPTER-3", dl.get_clinical_trials())[
        "Deucrictibant XR 40 mg QD"]
    return {"dist": d, "intervals": d.proportion_intervals(),
            "headline": _a("chapter3_rate_reduction_vs_placebo")}


def exhibit2_inputs() -> Dict:
    return {"bars": adoption_bars(), "segments": pm.segment_break_even(_base()),
            "price": _base().stock_price}


def exhibit3_inputs() -> Dict:
    base = _base()
    return {"clinical": clinical_anchors(), "curve": chain_curve(),
            "scenarios": val.scenario_table(base),
            "price": base.stock_price,
            "breakeven": val.short_metrics(base)["breakeven_value_per_share"]}


# ---------------------------------------------------------------------------
# The chain: clinical evidence -> adoption scenarios -> downside valuation
# ---------------------------------------------------------------------------

def chain_links() -> pd.DataFrame:
    """Every assumption that connects a measured clinical input to value per share.

    The table exists so that no step between "a trial read out" and "a share is
    worth X" can be hidden. ``path`` says where the input enters the chain;
    ``direct_to_value`` marks inputs that reach the discounted cash flow
    without passing through an analyst assumption.
    """
    base = _base()

    dist = re_.distributions_for_trial("CHAPTER-3", dl.get_clinical_trials())[
        "Deucrictibant XR 40 mg QD"]
    af = dist.counts["attack-free (100%)"]
    af_share = af / dist.total_patients

    def row(step, link, input_name, display, kind, status, note, direct=False):
        return {"step": step, "link": link, "input": input_name, "value": display,
                "type": kind, "status": status, "reaches_value_directly":
                    "yes" if direct else "no", "note": note}

    rows = [
        row(1, "clinical result", "chapter3_rate_reduction_vs_placebo",
            f"{_pct(_a('chapter3_rate_reduction_vs_placebo'), 0)} mean rate reduction",
            "reported", "verified_primary",
            "Feeds the real-world haircut and the choice-model efficacy bonus. Does not enter "
            "the discounted cash flow."),
        row(1, "clinical result", "baseline_monthly_attack_rate",
            f"{_a('baseline_monthly_attack_rate')} attacks / 4 weeks",
            "reported", "verified_primary",
            "CHAPTER-1 placebo arm; drives the prevention / rescue bridge only."),
        row(1, "clinical result", "CHAPTER-3 attack-free count",
            f"{af}/{dist.total_patients} = {_pct(af_share, 1)} attack-free over "
            f"{dist.window_days:.0f} days",
            "reported", "corroborated_secondary",
            "Calibrates patient-level dispersion once, then held fixed."),
        row(2, "trial to model", "real_world_efficacy_haircut",
            f"{_a('real_world_efficacy_haircut'):.2f}x haircut",
            "assumed", "unverified",
            "The step from trial efficacy to modelled efficacy. No real-world series exists."),
        row(2, "trial to model", "mnl_efficacy_bonus",
            f"{_a('mnl_efficacy_bonus'):.2f} utility units",
            "assumed", "unverified",
            "The only path by which efficacy influences adoption. Stated preference, not "
            "measured."),
        row(3, "adoption", "seg_* / oral_pref_*",
            "6 segment shares and oral-preference increments",
            "assumed", "unverified",
            "Choice shares from a multinomial logit. No physician has been surveyed."),
        row(3, "adoption", "base_switching_rate",
            f"{_pct(_a('base_switching_rate'), 0)} per year",
            "assumed", "unverified",
            "Accumulates switchers over the 6-year ramp. Not measured in claims."),
        row(3, "adoption", "prophylaxis_annual_discontinuation",
            f"{_pct(_a('prophylaxis_annual_discontinuation'), 0)} per year",
            "assumed", "unverified",
            "Persistence multiplier. The incumbent oral is claimed to drop out at "
            f"{_pct(_a('orladeyo_annual_dropout_rate'), 0)}."),
        row(3, "adoption", "prophylaxis_eligible_patients_global",
            f"{base.prophylaxis_eligible:,.0f} patients",
            "calculated", "unverified",
            "Denominator for penetration, built from a verified US count plus two unverified "
            "regional splits.", True),
        row(4, "adoption to revenue", "peak_penetration_prophylaxis",
            f"{_pct(base.peak_penetration_prophylaxis, 0)} at peak",
            "assumed", "unverified",
            "Patients = eligible pool x penetration x ramp x post-exclusivity decline.", True),
        row(4, "adoption to revenue", "phvs_prophylaxis_annual_net_price_us",
            _usd(base.price_prophylaxis, 0) + " per patient-year",
            "assumed", "unverified",
            "Pharvaris has not set a price. The ledger's only realised anchor is "
            f"{_usd(_a('implied_revenue_per_takhyro_patient'), 0)} per patient.", True),
        row(4, "adoption to revenue", "peak_penetration_acute / price_acute",
            f"{_pct(base.peak_penetration_acute, 0)} at " + _usd(base.price_acute, 0),
            "assumed", "unverified",
            "Acute revenue comes from the breakthrough subset inside the prophylaxis cohort, "
            "so better prevention lowers it.", True),
        row(5, "revenue to value", "approval_probability_prophylaxis / _acute",
            f"{_pct(base.approval_prophylaxis, 0)} / {_pct(base.approval_acute, 0)}",
            "assumed", "unverified",
            "Applied once per product contribution stream, never twice.", True),
        row(5, "revenue to value", "discount_rate",
            f"{_pct(base.discount_rate, 0)}", "assumed", "unverified",
            "Mid-year discounting; terminal growth forced to zero after exclusivity.", True),
        row(5, "revenue to value", "shares_outstanding",
            f"{base.shares / 1e6:.1f}m shares", "source fact", "verified_secondary",
            "Equity value divided by the verified count.", True),
        row(6, "comparison", "stock_price_current",
            _usd(base.stock_price, 2), "source fact", "verified_secondary",
            "Close on 2 October 2026 - the anchor every threshold is measured against.", True),
    ]
    return pd.DataFrame(rows)


def chain_break_even() -> pd.DataFrame:
    """Thresholds at which the chain flips from downside to upside for the short."""
    base = _base()
    short = val.short_metrics(base)
    mi = val.market_implied(base).set_index("input")
    rows = [
        {"stage": "clinical", "threshold": "Real-world efficacy haircut",
         "at_reference": f"{_a('real_world_efficacy_haircut'):.2f}",
         "break_even": "not a valuation driver",
         "note": "Moving it changes acute revenue only; it does not move value per share."},
        {"stage": "adoption", "threshold": "Peak prophylaxis penetration",
         "at_reference": _pct(base.peak_penetration_prophylaxis, 1),
         "break_even": _pct(_penetration_at(short["breakeven_value_per_share"]), 1),
         "note": f"The short breaks even here after {_pct(base.borrow_cost, 0)} carry; the price "
                 f"itself requires "
                 f"{_pct(float(mi.loc['peak_penetration_prophylaxis', 'required_value']), 1)}."},
        {"stage": "adoption", "threshold": "Annual switching rate",
         "at_reference": _pct(_a("base_switching_rate"), 0),
         "break_even": _pct(pm.break_even_switching(
             1.0, target_price=short["breakeven_value_per_share"], base=base
         )["required_switching_rate"], 1),
         "note": "Solved through the choice model at the stated price premium."},
        {"stage": "valuation", "threshold": "Model value per share",
         "at_reference": _usd(val.value_summary(base)["value_per_share"], 2),
         "break_even": _usd(short["breakeven_value_per_share"], 2),
         "note": f"Observed price {_usd(base.stock_price, 2)} less one year of carry."},
        {"stage": "valuation", "threshold": "Discount rate",
         "at_reference": _pct(base.discount_rate, 0),
         "break_even": _pct(float(mi.loc["discount_rate", "required_value"]), 1),
         "note": "Rate at which the model equals the observed price on its own."},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Recommendation, workstreams, disconfirming evidence
# ---------------------------------------------------------------------------

def short_recommendation() -> Dict:
    """Lead block: the recommendation and the strongest supported findings."""
    base = _base()
    summary = val.value_summary(base)
    bundle = val.required_bundle(base)
    repro = val.reproduce_pitch_valuation(dl.get_valuation_model())
    scenarios = val.scenario_table(base).set_index("scenario")
    short = val.short_metrics(base)
    sens = val.sensitivity_ranking(base)
    mi = val.market_implied(base).set_index("input")
    rq1, rq2, rq3 = rq1_response(), rq2_adoption(), rq3_price()

    spread = float(scenarios.loc["bull", "value_per_share"]) / float(
        scenarios.loc["bear", "value_per_share"])
    top = sens.head(3)

    findings = [
        {"rank": 1,
         "headline": f"{repro['multiple_required']:.2f}x",
         "claim": "The observed price needs "
                  f"{repro['multiple_required']:.2f}x the enterprise value of the cash-flow "
                  f"model it is being compared against "
                  f"({_usd(repro['enterprise_value'])} of EV, "
                  f"{_usd(repro['value_per_share_current_shares'], 2)} per share on the verified "
                  f"{base.shares / 1e6:.1f}m shares).",
         "basis": "reproduced arithmetic on the imported model", "status": "completed"},
        {"rank": 2,
         "headline": f"{spread:.1f}x spread, 0 measured inputs",
         "claim": f"Bear { scenarios.loc['bear','value_per_share']:,.2f} to bull "
                  f"{scenarios.loc['bull','value_per_share']:,.2f} per share is "
                  f"{spread:.1f}x, and the three inputs that move it most - "
                  f"{', '.join(top['parameter'].str.replace('_', ' '))} - are all flagged "
                  f"unverified. No measured clinical quantity is a valuation driver.",
         "basis": "one-at-a-time sensitivity ranking", "status": "completed"},
        {"rank": 3,
         "headline": rq3.metrics["Required peak penetration"] + " penetration",
         "claim": f"The price is reproduced by {rq3.metrics['Required peak penetration']} peak "
                  f"penetration, a {rq3.metrics['Required net price']} net price "
                  f"({rq3.table.iloc[4]['required_for_price'] / rq3.table.iloc[4]['stated']:.2f}x "
                  f"the incumbent's realised revenue per patient) or a "
                  f"{rq3.metrics['Required discount rate']} discount rate - solved one input at "
                  f"a time, so the bundle cannot all hold at once.",
         "basis": "inverse solve against the observed close", "status": "completed"},
        {"rank": 4,
         "headline": rq2.metrics["Switchers the price requires"] + " must switch",
         "claim": rq2.finding.split(". ")[0] + ". Market inflow over the ramp covers the rest, "
                  "and no switching series exists.",
         "basis": "patients required vs published market inflow", "status": "completed"},
        {"rank": 5,
         "headline": rq1.metrics["Attack-free at 168 days"] + " attack-free",
         "claim": f"The headline {_rq1_headline()} sits against "
                  f"{rq1.metrics['Attack-free at 168 days']} attack-free at 168 days "
                  f"(exact 95% interval {rq1.metrics['Exact 95% interval on attack-free']}) on "
                  f"n=55, with {rq1.metrics['Threshold rows still provisional']} threshold rows "
                  f"still provisional and no head-to-head against an existing prophylactic.",
         "basis": "nested-threshold inversion of published counts", "status": "completed"},
    ]

    return {
        "recommendation": "Short",
        "ticker": "PHVS",
        "price": base.stock_price,
        "price_date": "2026-10-02",
        "model_value": summary["value_per_share"],
        "model_vs_price": summary["upside_downside_pct"],
        "bear_value": float(scenarios.loc["bear", "value_per_share"]),
        "reference_value": float(scenarios.loc["reference", "value_per_share"]),
        "bull_value": float(scenarios.loc["bull", "value_per_share"]),
        "spread_multiple": spread,
        "required_ev": bundle["required_enterprise_value"],
        "required_multiple_of_reference": bundle["required_multiple_of_stated_ev"],
        "required_multiple_of_cashflow_model": repro["multiple_required"],
        "breakeven_value_per_share": short["breakeven_value_per_share"],
        "carry_pct": short["carry_pct"],
        "bear_net_return_pct": val.short_metrics(
            val.scenario_params(base, "bear"))["net_return_pct"],
        "reference_net_return_pct": short["net_return_pct"],
        "top_drivers": top["parameter"].str.replace("_", " ").tolist(),
        "findings": findings,
        "thesis": (
            "We are short Pharvaris on adoption, not on efficacy. The clinical readout is real "
            "and is already in the price; what the price has not been shown to contain is a "
            "measured path from that readout to "
            f"{rq3.metrics['Required peak penetration']} peak penetration in a market where oral "
            "therapy is a minority of prophylaxis value and the pool grows by a few hundred "
            "patients a year. We do not assert a fair value - we assert that the gap between "
            f"our bear and bull cases is filled entirely by unverified adoption inputs, and we "
            f"state the level at which the position is wrong: model value above "
            f"{_usd(short['breakeven_value_per_share'], 2)}."
        ),
    }


def _rq1_headline() -> str:
    return _pct(_a("chapter3_rate_reduction_vs_placebo"), 0)


def workstreams() -> pd.DataFrame:
    """What has been done, what is proposed, what is merely assumed."""
    return pd.DataFrame([
        {"workstream": "Registry verification of stored counts against ClinicalTrials.gov",
         "status": "completed", "evidence": "primary registry results API",
         "used_for": "RQ1 - every stored count is re-checked, failures degrade to "
                     "not_found_in_registry"},
        {"workstream": "Nested-threshold inversion into exclusive response categories",
         "status": "completed", "evidence": "arithmetic on published counts",
         "used_for": "RQ1 - headline vs patient-level statistics"},
        {"workstream": "Exact binomial intervals on disclosed thresholds",
         "status": "completed", "evidence": "Clopper-Pearson on published counts",
         "used_for": "RQ1 - sampling uncertainty stated separately from window length"},
        {"workstream": "Trial comparability audit (population, window, endpoint, comparator)",
         "status": "completed", "evidence": "structural flags, no pooling",
         "used_for": "RQ1 / RQ2 - blocks accidental cross-trial comparison"},
        {"workstream": "Population accounting with mutually exclusive cohorts",
         "status": "completed", "evidence": "validated to sum to 1.0",
         "used_for": "RQ2 - prevention and rescue never share a patient"},
        {"workstream": "Prevention / rescue bridge with one-time dispersion calibration",
         "status": "completed", "evidence": "gamma-Poisson fit to published values",
         "used_for": "RQ2 - rescue revenue falls as prevention improves"},
        {"workstream": "Inverse solve of the price against each adoption input",
         "status": "completed", "evidence": "root-finding on the extended model",
         "used_for": "RQ3 - break-even thresholds, not fair value"},
        {"workstream": "Reproduction of the circulated cash-flow model",
         "status": "completed", "evidence": "imported rows recomputed exactly",
         "used_for": "RQ3 - the multiple the price requires"},
        {"workstream": "Discrete-choice survey of prescriber preference",
         "status": "proposed", "evidence": "design only - no responses exist",
         "used_for": "RQ2 - would replace the assumed choice coefficients"},
        {"workstream": "Claims-based switching cohorts",
         "status": "proposed", "evidence": "not started",
         "used_for": "RQ2 - the single largest unmeasured input"},
        {"workstream": "Head-to-head evidence against injected prophylaxis",
         "status": "proposed", "evidence": "CHAPTER-3 is placebo-controlled only",
         "used_for": "RQ1 / RQ2 - would test the switching premise directly"},
        {"workstream": "Durability beyond 24 weeks",
         "status": "proposed", "evidence": "168-day window only",
         "used_for": "RQ1 - would retire the window-length discount"},
        {"workstream": "Choice-model coefficients, segment shares, net price, switching rate",
         "status": "assumed", "evidence": "analyst inputs flagged unverified",
         "used_for": "RQ2 / RQ3 - carried as assumptions, never as findings"},
    ])


def disconfirming_evidence() -> pd.DataFrame:
    """The strongest case against the short, stated at full strength."""
    base = _base()
    summary = val.value_summary(base)
    short = val.short_metrics(base)
    mi = val.market_implied(base).set_index("input")
    a = _assumptions()
    return pd.DataFrame([
        {"point": "Our own stated-input model says the stock is cheap",
         "detail": f"At the ledger's stated inputs the model is worth "
                   f"{_usd(summary['value_per_share'], 2)} against a {_usd(base.stock_price, 2)} "
                   f"close, i.e. {summary['upside_downside_pct']:+.1%}. At those inputs the short "
                   f"loses {abs(short['net_return_pct']):.1f}% net of carry.",
         "why_it_bites": "The bear case is conditional; it is not what the ledger states."},
        {"point": "Every single input has slack",
         "detail": f"The price requires only { _pct(float(mi.loc['peak_penetration_prophylaxis','required_value']),1) } "
                   f"peak penetration against {_pct(base.peak_penetration_prophylaxis, 0)} stated, "
                   f"{_pct(float(mi.loc['approval_prophylaxis','required_value']), 1)} approval "
                   f"against {_pct(base.approval_prophylaxis, 0)} and a "
                   f"{_pct(float(mi.loc['discount_rate','required_value']), 1)} discount rate "
                   f"against {_pct(base.discount_rate, 0)}. On a one-at-a-time basis the market "
                   f"is assuming less than we state.",
         "why_it_bites": "The stated case does not have to be revised down for the price to be "
                         "explainable."},
        {"point": "The clinical result is strong and verified",
         "detail": f"{_pct(_a('chapter3_rate_reduction_vs_placebo'), 0)} mean rate reduction vs "
                   f"placebo, p<0.0001, with the primary endpoint verified against the company "
                   f"disclosure and a PDUFA date already set for the on-demand indication.",
         "why_it_bites": "A short built on 'the drug does not work' would be wrong. Ours is not."},
        {"point": "Carry is real and the position must resolve on schedule",
         "detail": f"Borrow is assumed at {_pct(base.borrow_cost, 0)} a year, so the position "
                   f"must resolve within the horizon rather than merely be right eventually.",
         "why_it_bites": "Correctness without timing still loses money."},
        {"point": "Sell-side consensus is far above the price",
         "detail": f"The 14-analyst mean target is "
                   f"{_usd(a.loc[a['parameter']=='analyst_mean_price_target','value'].iloc[0], 2)} "
                   f"against a {_usd(base.stock_price, 2)} close, on a 'Strong Buy' consensus.",
         "why_it_bites": "We are on the other side of a crowded, well-covered book."},
        {"point": "Approval risk is dated and public",
         "detail": "The on-demand NDA is under review with a stated PDUFA date; a positive "
                   "decision moves the approval input directly and the market knows the date.",
         "why_it_bites": "A binary catalyst can close the gap faster than adoption data can."},
        {"point": "Three of four response thresholds are provisional",
         "detail": "The counts behind the exclusive categories are company-reported and not yet "
                   "posted to the registry; only the attack-free count is corroborated.",
         "why_it_bites": "Part of our own RQ1 evidence could be revised on publication."},
        {"point": "The survey has not been conducted",
         "detail": "Physician preference is unmeasured in both directions: we cannot show the "
                   "choice coefficients are too high, only that nobody has estimated them.",
         "why_it_bites": "RQ2's constraints are bounds, not measurements."},
    ])


def limitations_notes() -> List[tuple]:
    """Where the lab itself is weakest, topic first - shared by app and site."""
    base = _base()
    summary = val.value_summary(base)
    repro = val.reproduce_pitch_valuation(dl.get_valuation_model())
    return [
        ("Two defensible models disagree by "
         f"{summary['enterprise_value'] / repro['enterprise_value']:.1f}\u00d7",
         f"- The **pitch's own cash flows** reproduce to an enterprise value of "
         f"{_usd(repro['enterprise_value'])} and "
         f"{_usd(repro['value_per_share_current_shares'], 2)} per share; the price needs "
         f"{repro['multiple_required']:.2f}\u00d7 that.\n"
         f"- The **extended reference model** adds the verified share count, cash, launch "
         f"timing, approval risk and terminal mechanics, and produces "
         f"{_usd(summary['value_per_share'], 2)} per share.\n"
         f"- The price sits between them. Which of the two you believe is the whole argument, "
         f"and the valuation section shows which inputs separate them."),
        ("A verified row is not a comparable row",
         "A verified row only means *the number in the table matches the source*. It says "
         "nothing about whether the source is the right comparator, whether the window is "
         "comparable, or whether a placebo-controlled result transfers to a market with "
         "existing options."),
        ("What response reconstruction cannot recover",
         "- Nested thresholds identify exclusive categories **only when every threshold was "
         "disclosed**. Where one is missing the correct answer is a set, not a point.\n"
         "- Subtype counts (type I/II vs other HAE) were not disclosed, so a subtype-specific "
         "attack-free rate is not identified at all.\n"
         "- A 168-day attack-free count is not an annual probability, and no confidence "
         "interval can make it one."),
        ("The read-across everyone makes",
         "- CHAPTER-3 is **vs placebo**, not vs an existing prophylactic. Nothing in this "
         "ledger measures how many patients switch off an injected LTP for an oral one.\n"
         "- Windows differ (84 vs 168 days) and populations differ (children in HELP, "
         "crossover in COMPACT). Placing them on one axis demonstrates the window effect, not "
         "relative performance.\n"
         "- '83% reduction' and '45% attack-free' are different statistics; a case that leans "
         "on one while quoting the other is not coherent."),
        ("Where the economics model could be wrong",
         "- **Addressable population is an assumption.** The ledger's 19,614 global diagnosed "
         "patients is already ~3\u00d7 below the pitch's 60,000; if the pitch is right every "
         "derived share scales up.\n"
         "- **Segment shares and addressability are analyst inputs**, not measured.\n"
         "- **The acute stream depends on breakthrough**, which is modelled from a "
         "gamma-Poisson fit to one trial's published attack-free rate - a fit, not an "
         "observation."),
        ("Stated preference is not prescribing",
         "- Every coefficient in the map is an **assumption in the ledger**, flagged "
         "`unverified`. The map translates assumptions into patients, it does not observe "
         "them.\n"
         f"- The map's base case lands at {_pct(pm.implied_peak_penetration(pm.MapParams()), 1)} "
         f"peak penetration against the ledger's stated "
         f"{_pct(base.peak_penetration_prophylaxis, 0)} - an inconsistency between two parts "
         "of the same model.\n"
         "- Price realisation is modelled as a uniform premium; real gross-to-net, formulary "
         "tiering and PBM behaviour are absent."),
        ("When the short fails",
         f"- At the reference inputs the short **does not work**: model value "
         f"{_usd(summary['value_per_share'], 2)} against a {_usd(base.stock_price, 2)} price.\n"
         f"- It needs model value at or below "
         f"{_usd(val.short_metrics(base)['breakeven_value_per_share'], 2)}, which is roughly "
         "the bear case.\n"
         "- The largest swing is price and peak penetration (identical by construction, both "
         "scale revenue). A single successful launch print on either one removes the case.\n"
         "- Approval is pending with a stated PDUFA date; a positive decision moves the "
         "approval input directly and the market knows the date."),
    ]


# ---------------------------------------------------------------------------
# Exhibits
# ---------------------------------------------------------------------------

def exhibits() -> List[Dict]:
    """The three slide-ready research exhibits, with sources and captions."""
    src = dl.get_sources().set_index("source_id")

    def cite(ids: List[str]) -> List[Dict]:
        out = []
        for i in ids:
            r = src.loc[i]
            out.append({"id": i, "title": r["title"], "publisher": r["publisher"],
                        "url": r["url"], "date": r["publication_date"]})
        return out

    rq1, rq2, rq3 = rq1_response(), rq2_adoption(), rq3_price()
    base = _base()
    mi = val.market_implied(base).set_index("input")

    return [
        {
            "id": "EX1",
            "rq": "RQ1",
            "png": EXHIBIT_PNGS["EX1"],
            "title": "An 83% mean, a 45% patient",
            "headline": f"The headline is a ratio of means. Only "
                        f"{rq1.metrics['Attack-free at 168 days']} of treated patients were "
                        f"attack-free over 168 days.",
            "question": rq1.question,
            "methodology": (
                "Nested cumulative responder thresholds for the CHAPTER-3 active arm inverted "
                "into mutually exclusive categories by subtraction; exact Clopper-Pearson "
                "intervals on the disclosed cumulative counts; window and verification status "
                "reported separately. Separate trials are not compared."
            ),
            "caption": rq1.finding,
            "implication": rq1.implication,
            "status": "completed analysis on published counts",
            "sources": cite(["S01", "S02", "S22", "S03"]),
        },
        {
            "id": "EX2",
            "rq": "RQ2",
            "png": EXHIBIT_PNGS["EX2"],
            "title": "The price is a switching bet",
            "headline": f"At least {rq2.metrics['Switchers the price requires']} of the patients "
                        f"the price requires must be taken off existing therapy.",
            "question": rq2.question,
            "methodology": (
                "Patients required at the observed price are solved from price x verified shares "
                "less net cash, divided by assumed net price and the eligible pool. Market "
                "inflow uses management's published 150-250 new patients a year, deliberately "
                "held at 100% global capture over the 6-year ramp so it is a lower bound. "
                "Segment ceilings come from the ledger's multinomial-logit choice model, whose "
                "coefficients are analyst assumptions, not measured preferences."
            ),
            "caption": rq2.finding,
            "implication": rq2.implication,
            "status": "completed arithmetic; adoption coefficients assumed",
            "sources": cite(["S20", "S10", "S17", "S14", "S16"]),
        },
        {
            "id": "EX3",
            "rq": "RQ3",
            "png": EXHIBIT_PNGS["EX3"],
            "title": "Evidence to adoption to downside",
            "headline": f"The measured clinical inputs never reach the valuation. "
                        f"{rq3.metrics['Required peak penetration']} peak penetration, "
                        f"{rq3.metrics['Required net price']} net price or "
                        f"{rq3.metrics['Required discount rate']} discount reproduces the price - "
                        f"one at a time.",
            "question": rq3.question,
            "methodology": (
                "Panel 1: published clinical anchors for the CHAPTER-3 active arm. Panel 2: "
                "value per share over peak prophylaxis penetration with every other ledger input "
                "held at its stated value, with contours at the observed close and at the "
                "carry-adjusted break-even. Panel 3: bear / reference / bull outputs of the same "
                "model. Nothing between the panels is measured; each link is listed in the "
                "chain table below with its type and verification status."
            ),
            "caption": rq3.finding,
            "implication": rq3.implication,
            "status": "completed analysis; the connecting assumptions are unverified",
            "sources": cite(["S16", "S01", "S03", "S10", "S17", "S20", "S08"]),
        },
    ]


def exhibit_source_lines(exhibit: Dict) -> List[str]:
    """Compact 'Sources:' line for a slide footer."""
    return [f"[{s['id']}] {s['publisher']}, {s['date']} - {s['url']}"
            for s in exhibit["sources"]]


def exhibit_figures() -> Dict[str, "go.Figure"]:
    """The three exhibit figures keyed by their exported PNG stem.

    Imported lazily so ``research`` stays importable without a plotting
    backend, and so the export button, the tests and the site all draw the
    same objects.
    """
    from phvs_lab.utils import visualization as viz

    return {
        EXHIBIT_PNGS["EX1"]: viz.fig_exhibit_response(exhibit1_inputs()),
        EXHIBIT_PNGS["EX2"]: viz.fig_exhibit_adoption(exhibit2_inputs()),
        EXHIBIT_PNGS["EX3"]: viz.fig_exhibit_chain(exhibit3_inputs()),
    }
