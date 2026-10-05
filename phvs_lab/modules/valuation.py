"""Valuation: reproduce the pitch's model, then extend it.

Three distinct objects, never mixed:

1. ``reproduce_pitch_valuation`` - the imported cash-flow CSV, recomputed
   exactly, so its arithmetic and its stale inputs can be checked.
2. ``ValuationParams`` / ``build_cashflows`` - an editable risk-adjusted DCF
   driven by the assumption ledger.  Approval probability is applied **once**,
   to each product's contribution stream; adoption is expressed as peak
   penetration inside that stream, so the two risks are not multiplied twice.
3. ``market_implied`` - the inverse problem: what must be true for the observed
   price?

Every output is a scenario result, not a forecast or a consensus estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from phvs_lab.modules.data_loader import (
    get_assumption,
    get_commercial_assumptions,
    get_valuation_model,
)

SCENARIOS = ("bear", "reference", "bull")


class ValuationError(ValueError):
    """Raised for infeasible or inconsistent valuation inputs."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class ValuationParams:
    """All editable inputs of the extended model."""

    # populations (denominators)
    prophylaxis_eligible: float = 17_653.0
    acute_eligible: float = 19_614.0

    # launch / uptake
    launch_year_prophylaxis: int = 2028
    launch_year_acute: int = 2027
    peak_penetration_prophylaxis: float = 0.15
    peak_penetration_acute: float = 0.08
    time_to_peak_prophylaxis: float = 6.0
    time_to_peak_acute: float = 4.0

    # price / margin
    price_prophylaxis: float = 425_000.0
    price_acute: float = 85_000.0
    cogs_prophylaxis: float = 0.12
    cogs_acute: float = 0.15

    # operating cost
    rd_pre_launch: float = 140_000_000.0
    rd_post_launch: float = 70_000_000.0
    rd_growth: float = 0.03
    sga_base: float = 63_200_000.0
    sga_peak: float = 150_000_000.0

    # capital intensity / tax
    capex_pct: float = 0.02
    nwc_pct: float = 0.05
    tax_rate: float = 0.18

    # exclusivity / horizon
    loe_year: int = 2042
    post_loe_decline: float = 0.15
    rd_post_loe_floor: float = 0.50   # R&D is cut on LOE but never below this share
    forecast_end_year: int = 2049

    # discounting
    discount_rate: float = 0.10
    terminal_growth: float = 0.02

    # approval risk (applied once, per product)
    approval_prophylaxis: float = 0.88
    approval_acute: float = 0.92

    # capital structure
    net_cash: float = 362_860_000.0
    shares: float = 70_200_000.0
    annual_dilution: float = 0.05
    diluted_share_years: float = 5.0

    # market / short
    stock_price: float = 31.19
    borrow_cost: float = 0.06

    def validate(self) -> None:
        if self.discount_rate <= 0:
            raise ValuationError("discount_rate must be positive.")
        if self.terminal_growth >= self.discount_rate:
            raise ValuationError(
                f"terminal_growth ({self.terminal_growth:.3f}) must be below "
                f"discount_rate ({self.discount_rate:.3f}) for a finite terminal value."
            )
        if self.terminal_growth < 0:
            raise ValuationError("terminal_growth must be non-negative.")
        for name in ("peak_penetration_prophylaxis", "peak_penetration_acute",
                     "approval_prophylaxis", "approval_acute", "cogs_prophylaxis",
                     "cogs_acute", "tax_rate", "post_loe_decline"):
            v = getattr(self, name)
            if not 0 <= v <= 1:
                raise ValuationError(f"{name} must lie in [0, 1]; got {v}.")
        if self.shares <= 0 or self.prophylaxis_eligible < 0 or self.acute_eligible < 0:
            raise ValuationError("Shares and populations must be positive.")
        if self.launch_year_acute >= self.forecast_end_year or \
                self.launch_year_prophylaxis >= self.forecast_end_year:
            raise ValuationError(
                "Launch years must precede forecast_end_year or the model has no "
                "commercial period to value."
            )
        if self.time_to_peak_prophylaxis <= 0 or self.time_to_peak_acute <= 0:
            raise ValuationError("time_to_peak must be positive.")
        if self.price_prophylaxis <= 0 or self.price_acute <= 0:
            raise ValuationError("Net prices must be positive.")
        if self.loe_year < self.launch_year_acute:
            raise ValuationError("loe_year cannot precede the first launch year.")

    @property
    def diluted_shares(self) -> float:
        return self.shares * (1 + self.annual_dilution) ** self.diluted_share_years


def params_from_assumptions(df: Optional[pd.DataFrame] = None) -> ValuationParams:
    """Build model inputs from the assumption ledger (every value traceable)."""
    df = df if df is not None else get_commercial_assumptions()

    def a(name: str) -> float:
        return get_assumption(name, df)

    return ValuationParams(
        prophylaxis_eligible=a("prophylaxis_eligible_patients_global"),
        acute_eligible=a("acute_eligible_patients_global"),
        launch_year_prophylaxis=int(a("phvs_prophylaxis_launch_year")),
        launch_year_acute=int(a("phvs_acute_launch_year")),
        peak_penetration_prophylaxis=a("peak_penetration_prophylaxis"),
        peak_penetration_acute=a("peak_penetration_acute"),
        time_to_peak_prophylaxis=a("prophylaxis_time_to_peak_years"),
        time_to_peak_acute=a("acute_time_to_peak_years"),
        price_prophylaxis=a("phvs_prophylaxis_annual_net_price_us"),
        price_acute=a("phvs_acute_annual_net_price_us"),
        cogs_prophylaxis=a("cogs_prophylaxis_pct_revenue"),
        cogs_acute=a("cogs_acute_pct_revenue"),
        rd_pre_launch=a("rd_expense_2026_usd"),
        rd_post_launch=a("rd_post_approval_usd"),
        rd_growth=a("rd_growth_post_launch"),
        sga_base=a("sganda_expense_2026_usd"),
        sga_peak=a("sganda_peak_usd"),
        capex_pct=a("capex_pct_revenue"),
        nwc_pct=a("nwc_pct_incremental_revenue"),
        tax_rate=a("effective_tax_rate"),
        loe_year=int(a("loe_year")),
        post_loe_decline=a("post_loe_annual_decline"),
        forecast_end_year=int(a("forecast_end_year")),
        discount_rate=a("discount_rate"),
        terminal_growth=a("terminal_growth_rate"),
        approval_prophylaxis=a("approval_probability_prophylaxis"),
        approval_acute=a("approval_probability_acute"),
        net_cash=a("cash_and_investments_usd") - a("total_debt_usd"),
        shares=a("shares_outstanding"),
        annual_dilution=a("annual_dilution_rate"),
        stock_price=a("stock_price_current"),
        borrow_cost=a("borrow_cost_annual"),
    )


# ---------------------------------------------------------------------------
# Cash flows
# ---------------------------------------------------------------------------

def _ramp(years: np.ndarray, launch: int, ttp: float) -> np.ndarray:
    """Linear uptake from launch to peak, zero before launch."""
    return np.clip((years - launch + 1.0) / ttp, 0.0, 1.0)


def build_cashflows(p: ValuationParams) -> pd.DataFrame:
    """Annual risk-adjusted cash flows with product-level attribution.

    Columns keep every component visible: revenue, gross profit, operating cost,
    tax, capex, working capital, the fixed stream, each product's after-tax
    contribution, the discount factor and the present values.
    """
    p.validate()
    years = np.arange(2026, p.forecast_end_year + 1)
    t = years - 2026

    ramp_p = _ramp(years, p.launch_year_prophylaxis, p.time_to_peak_prophylaxis)
    ramp_a = _ramp(years, p.launch_year_acute, p.time_to_peak_acute)
    decline = np.where(years > p.loe_year,
                       (1 - p.post_loe_decline) ** (years - p.loe_year), 1.0)

    patients_p = p.prophylaxis_eligible * p.peak_penetration_prophylaxis * ramp_p * decline
    patients_a = p.acute_eligible * p.peak_penetration_acute * ramp_a * decline
    rev_p = patients_p * p.price_prophylaxis
    rev_a = patients_a * p.price_acute
    revenue = rev_p + rev_a

    gross_p = rev_p * (1 - p.cogs_prophylaxis)
    gross_a = rev_a * (1 - p.cogs_acute)

    first_launch = min(p.launch_year_prophylaxis, p.launch_year_acute)
    rd_post_path = p.rd_post_launch * (1 + p.rd_growth) ** np.maximum(years - first_launch, 0)
    # research is cut once exclusivity is lost, but never below the stated floor
    rd_post_path = rd_post_path * np.maximum(decline, p.rd_post_loe_floor)
    rd = np.where(years < first_launch, p.rd_pre_launch, rd_post_path)
    ramp_max = np.maximum(ramp_p, ramp_a)
    # the commercial build-out shrinks again after exclusivity is lost
    sga = p.sga_base + (p.sga_peak - p.sga_base) * ramp_max * decline

    ebit = gross_p + gross_a - rd - sga

    # tax with loss carry-forward
    tax = np.zeros_like(ebit)
    nol = 0.0
    for i, e in enumerate(ebit):
        if e <= 0:
            nol += -e
            tax[i] = 0.0
        else:
            shield = min(nol, e)
            nol -= shield
            tax[i] = p.tax_rate * (e - shield)
    tau = np.divide(tax, ebit, out=np.zeros_like(tax), where=ebit > 0)

    capex = p.capex_pct * revenue
    prev_rev = np.concatenate([[0.0], revenue[:-1]])
    delta_nwc = p.nwc_pct * (revenue - prev_rev)

    # attribution of the revenue-linked items (exactly additive)
    with np.errstate(invalid="ignore", divide="ignore"):
        share_p = np.where(revenue > 0, rev_p / np.where(revenue > 0, revenue, 1.0), 0.0)
        share_a = np.where(revenue > 0, rev_a / np.where(revenue > 0, revenue, 1.0), 0.0)
    sga_increment = sga - p.sga_base
    contrib_p = (gross_p - capex * share_p - delta_nwc * share_p) * (1 - tau)
    contrib_a = (gross_a - capex * share_a - delta_nwc * share_a) * (1 - tau)

    # Cost treatment: pre-launch research and baseline overhead exist whatever
    # happens, so they are carried at face value.  Post-launch research and the
    # commercial scale-up only exist if at least one indication is approved, so
    # they are weighted by P(either approval).  Approval is still applied once
    # per product's contribution stream above, never twice to the same cash flow.
    p_either = 1 - (1 - p.approval_prophylaxis) * (1 - p.approval_acute)
    pre_launch = years < first_launch
    certain = -np.where(pre_launch, rd, 0.0) - p.sga_base
    conditional = -np.where(~pre_launch, rd, 0.0) - sga_increment
    fixed = certain + conditional * p_either

    fcf_full = ebit - tax - capex - delta_nwc
    fcf_risk_adjusted = (fixed + p.approval_prophylaxis * contrib_p
                         + p.approval_acute * contrib_a)

    df = 1.0 / (1 + p.discount_rate) ** (t + 0.5)   # mid-year convention
    g_applied = 0.0 if years[-1] > p.loe_year else p.terminal_growth
    last_risk_adjusted = fcf_risk_adjusted[-1]
    terminal = 0.0
    if last_risk_adjusted > 0 and p.discount_rate > g_applied:
        terminal = last_risk_adjusted * (1 + g_applied) / (p.discount_rate - g_applied)

    out = pd.DataFrame({
        "year": years,
        "revenue_prophylaxis": rev_p,
        "revenue_acute": rev_a,
        "revenue_total": revenue,
        "patients_prophylaxis": patients_p,
        "patients_acute": patients_a,
        "gross_profit": gross_p + gross_a,
        "rd": rd,
        "sganda": sga,
        "ebitda": gross_p + gross_a - rd - sga,
        "tax": tax,
        "effective_tax_rate": tau,
        "capex": capex,
        "change_nwc": delta_nwc,
        "fcf_approved_path": fcf_full,
        "fixed_stream": fixed,
        "contribution_prophylaxis": contrib_p,
        "contribution_acute": contrib_a,
        "fcf_risk_adjusted": fcf_risk_adjusted,
        "discount_factor": df,
        "pv_fixed": df * fixed,
        "pv_prophylaxis": df * p.approval_prophylaxis * contrib_p,
        "pv_acute": df * p.approval_acute * contrib_a,
        "pv_fcf": df * fcf_risk_adjusted,
    })
    out.attrs["terminal_value"] = float(terminal)
    out.attrs["pv_terminal"] = float(terminal * df[-1])
    out.attrs["terminal_growth_applied"] = float(g_applied)
    return out


def value_summary(p: ValuationParams) -> Dict[str, float]:
    """Enterprise, equity and per-share value under the given inputs."""
    cf = build_cashflows(p)
    pv_operating = float(cf["pv_fcf"].sum())
    pv_terminal = float(cf.attrs["pv_terminal"])
    enterprise = pv_operating + pv_terminal
    equity = enterprise + p.net_cash
    return {
        "pv_operating": pv_operating,
        "pv_terminal": pv_terminal,
        "terminal_value": float(cf.attrs["terminal_value"]),
        "terminal_growth_applied": float(cf.attrs["terminal_growth_applied"]),
        "enterprise_value": enterprise,
        "net_cash": p.net_cash,
        "equity_value": equity,
        "value_per_share": equity / p.shares,
        "value_per_diluted_share": equity / p.diluted_shares,
        "stock_price": p.stock_price,
        "upside_downside": equity / p.shares - p.stock_price,
        "upside_downside_pct": (equity / p.shares) / p.stock_price - 1,
        "implied_market_cap": p.stock_price * p.shares,
        "required_ev_for_price": p.stock_price * p.shares - p.net_cash,
        "terminal_share_of_ev": pv_terminal / enterprise if enterprise else np.nan,
    }


# ---------------------------------------------------------------------------
# Reproducing the pitch model
# ---------------------------------------------------------------------------

def reproduce_pitch_valuation(df: Optional[pd.DataFrame] = None) -> Dict:
    """Recompute the imported pitch cash-flow model and reconcile it to the price."""
    df = df if df is not None else get_valuation_model()
    numeric_year = pd.to_numeric(df["year"], errors="coerce")
    explicit = df[numeric_year.notna()].copy()
    terminal_row = df[numeric_year.isna()]

    explicit_pv = float(explicit["pv_fcf_usd"].sum())
    terminal_pv = float(terminal_row["pv_fcf_usd"].sum()) if not terminal_row.empty else 0.0
    stored_cumulative = float(explicit["cumulative_pv_usd"].iloc[-1])
    fcf_sum = float(explicit["fcf_usd"].sum())

    # discount rate implied by the first row (checks the model's own arithmetic)
    first = explicit.iloc[0]
    implied_rate = np.nan
    if first["fcf_usd"] != 0:
        implied_rate = float(first["fcf_usd"] / first["pv_fcf_usd"] - 1)

    a = get_commercial_assumptions()
    net_cash = get_assumption("cash_and_investments_usd", a) - get_assumption("total_debt_usd", a)
    shares_now = get_assumption("shares_outstanding", a)
    price_now = get_assumption("stock_price_current", a)
    shares_pitch = float(explicit["shares_outstanding"].iloc[0])

    ev = explicit_pv + terminal_pv
    equity_now = ev + net_cash

    findings: List[Dict] = [
        {"check": "cumulative column excludes terminal value",
         "stored": stored_cumulative, "recomputed_with_terminal": explicit_pv + terminal_pv,
         "message": ("The pitch's cumulative_pv_usd column stops at the explicit forecast and "
                     "omits the terminal row, so the headline present value understates the "
                     f"model by {terminal_pv:,.0f} USD of PV.")},
        {"check": "share count", "stored": shares_pitch, "current": shares_now,
         "message": (f"The model is built on {shares_pitch/1e6:.1f}m shares; verified "
                     f"outstanding shares are {shares_now/1e6:.1f}m "
                     f"({shares_now/shares_pitch:.2f}x higher), so any per-share figure in the "
                     f"pitch is stale.")},
        {"check": "share price input", "stored": get_assumption("stock_price_pitch_deck", a),
         "current": price_now,
         "message": "The embedded price is the pitch's own reference, not today's close."},
        {"check": "implied discount rate", "stored": implied_rate, "current": np.nan,
         "message": (f"The first-year row implies a discount rate of {implied_rate:.1%}, "
                     f"consistent with a 10% model.")},
    ]

    return {
        "explicit_pv": explicit_pv,
        "terminal_pv": terminal_pv,
        "enterprise_value": ev,
        "net_cash": net_cash,
        "equity_value": equity_now,
        "shares_pitch": shares_pitch,
        "shares_current": shares_now,
        "value_per_share_pitch_shares": equity_now / shares_pitch,
        "value_per_share_current_shares": equity_now / shares_now,
        "stock_price": price_now,
        "required_equity_for_price": price_now * shares_now,
        "gap_to_price": price_now * shares_now - equity_now,
        "multiple_required": (price_now * shares_now - net_cash) / ev if ev else np.nan,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_params(base: ValuationParams, scenario: str) -> ValuationParams:
    """Bear / reference / bull parameter sets.

    Shifts are **multiplicative factors** applied to the stated inputs (rates
    are clipped to [0, 1]), so a scenario can never silently overwrite a level.
    """
    if scenario not in SCENARIOS:
        raise ValuationError(f"Unknown scenario '{scenario}'; expected one of {SCENARIOS}.")
    if scenario == "reference":
        return base

    factors = {
        "bear": dict(peak=0.70, price=0.90, approval=0.85, discount_add=0.015,
                     cost=1.05, decline=1.40),
        "bull": dict(peak=1.30, price=1.08, approval=1.06, discount_add=-0.01,
                     cost=0.97, decline=0.70),
    }[scenario]

    out = replace(
        base,
        peak_penetration_prophylaxis=min(base.peak_penetration_prophylaxis * factors["peak"], 1.0),
        peak_penetration_acute=min(base.peak_penetration_acute * factors["peak"], 1.0),
        price_prophylaxis=base.price_prophylaxis * factors["price"],
        price_acute=base.price_acute * factors["price"],
        approval_prophylaxis=min(base.approval_prophylaxis * factors["approval"], 0.99),
        approval_acute=min(base.approval_acute * factors["approval"], 0.99),
        discount_rate=max(base.discount_rate + factors["discount_add"], 0.01),
        sga_peak=base.sga_peak * factors["cost"],
        rd_post_launch=base.rd_post_launch * factors["cost"],
        post_loe_decline=min(base.post_loe_decline * factors["decline"], 0.60),
    )
    out.validate()
    return out


def scenario_table(base: Optional[ValuationParams] = None) -> pd.DataFrame:
    """Bear / reference / bull value summary."""
    base = base or params_from_assumptions()
    rows = []
    for s in SCENARIOS:
        p = scenario_params(base, s)
        v = value_summary(p)
        rows.append({"scenario": s, **v,
                     "vs_price_pct": v["upside_downside_pct"],
                     "peak_proph_patients": p.prophylaxis_eligible * p.peak_penetration_prophylaxis,
                     "peak_proph_revenue": p.prophylaxis_eligible * p.peak_penetration_prophylaxis
                     * p.price_prophylaxis,
                     "discount_rate": p.discount_rate,
                     "approval_prophylaxis": p.approval_prophylaxis})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The inverse problem: what must the market be assuming?
# ---------------------------------------------------------------------------

def _scalar_value(p: ValuationParams, field: str, x: float) -> float:
    return value_summary(replace(p, **{field: x}))["value_per_share"]


def market_implied(base: Optional[ValuationParams] = None,
                   price: Optional[float] = None,
                   fields: Tuple[str, ...] = ("peak_penetration_prophylaxis",
                                              "approval_prophylaxis",
                                              "discount_rate",
                                              "price_prophylaxis")
                   ) -> pd.DataFrame:
    """Solve, one input at a time, for the value that reproduces ``price``.

    Each row reports the level of a single input that would make the model
    equal the market price, holding everything else at its stated value, plus
    whether that level is even attainable.
    """
    base = base or params_from_assumptions()
    price = base.stock_price if price is None else price

    bounds = {
        "peak_penetration_prophylaxis": (1e-6, 1.0),
        "approval_prophylaxis": (1e-3, 1.0),
        "price_prophylaxis": (1_000.0, 3_000_000.0),
        # the discount rate can never be solved at or below terminal growth
        "discount_rate": (base.terminal_growth + 0.005, 0.60),
    }
    direction = {"peak_penetration_prophylaxis": "increasing",
                 "approval_prophylaxis": "increasing",
                 "price_prophylaxis": "increasing",
                 "discount_rate": "decreasing"}

    rows = []
    for field in fields:
        lo, hi = bounds[field]
        f = lambda x: _scalar_value(base, field, x) - price      # noqa: E731
        v_lo, v_hi = f(lo), f(hi)
        row = {"input": field, "stated_value": getattr(base, field),
               "market_price": price, "required_value": np.nan, "attainable": False,
               "stated_vs_required_pct": np.nan, "note": ""}
        increasing = direction[field] == "increasing"
        crosses = (v_lo <= 0 <= v_hi) if increasing else (v_hi <= 0 <= v_lo)
        if crosses:
            root = optimize.brentq(f, lo, hi, xtol=1e-10, rtol=1e-8)
            row["required_value"] = float(root)
            row["attainable"] = True
            stated = getattr(base, field)
            row["stated_vs_required_pct"] = (stated / root - 1) if root else np.nan
            row["note"] = ("Model matches the price when this input is set here, all else equal.")
        else:
            bound_hit = hi if (increasing and v_hi < 0) else lo
            row["required_value"] = float(bound_hit)
            row["note"] = (f"Even at the bound ({bound_hit:g}) the model does not reach "
                           f"${price:,.2f}; the price requires more than this input alone "
                           f"can provide.")
        rows.append(row)
    return pd.DataFrame(rows)


def required_bundle(base: Optional[ValuationParams] = None,
                    price: Optional[float] = None) -> Dict[str, float]:
    """Required enterprise value and the multiple of the stated model."""
    base = base or params_from_assumptions()
    price = base.stock_price if price is None else price
    stated = value_summary(base)
    required_equity = price * base.shares
    required_ev = required_equity - base.net_cash
    return {
        "price": price,
        "stated_value_per_share": stated["value_per_share"],
        "stated_equity_value": stated["equity_value"],
        "required_equity_value": required_equity,
        "required_enterprise_value": required_ev,
        "stated_enterprise_value": stated["enterprise_value"],
        "required_multiple_of_stated_ev": (required_ev / stated["enterprise_value"]
                                           if stated["enterprise_value"] else np.nan),
        "gap_equity": required_equity - stated["equity_value"],
    }


# ---------------------------------------------------------------------------
# Sensitivity ranking
# ---------------------------------------------------------------------------

SHOCKED_INPUTS = [
    "peak_penetration_prophylaxis", "peak_penetration_acute", "price_prophylaxis",
    "price_acute", "approval_prophylaxis", "approval_acute", "discount_rate",
    "rd_post_launch", "sga_peak", "tax_rate", "post_loe_decline", "shares",
]


def sensitivity_ranking(base: Optional[ValuationParams] = None,
                        shock: float = 0.10) -> pd.DataFrame:
    """Tornado table: value per share with each input shocked up and down."""
    base = base or params_from_assumptions()
    centre = value_summary(base)["value_per_share"]
    rows = []
    for field in SHOCKED_INPUTS:
        stated = getattr(base, field)
        try:
            up = value_summary(replace(base, **{field: stated * (1 + shock)}))["value_per_share"]
            down = value_summary(replace(base, **{field: stated * (1 - shock)}))["value_per_share"]
        except ValuationError:
            continue
        rows.append({"parameter": field, "stated_value": stated,
                     "value_at_plus": up, "value_at_minus": down,
                     "swing": abs(up - down),
                     "swing_pct_of_centre": abs(up - down) / centre if centre else np.nan,
                     "direction": "value rises with input" if up > down else
                                  "value falls with input"})
    out = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out.attrs["centre_value_per_share"] = centre
    return out


# ---------------------------------------------------------------------------
# Short metrics and invalidation
# ---------------------------------------------------------------------------

def short_metrics(base: Optional[ValuationParams] = None,
                  horizon_years: float = 1.0) -> Dict[str, float]:
    """Returns available to a short at the stated inputs (not advice)."""
    base = base or params_from_assumptions()
    value = value_summary(base)["value_per_share"]
    price = base.stock_price
    gross = (price - value) / price
    carry = base.borrow_cost * horizon_years
    net = gross - carry
    return {
        "stock_price": price,
        "value_per_share": value,
        "gross_return_pct": gross * 100,
        "annual_borrow_cost_pct": base.borrow_cost * 100,
        "carry_pct": carry * 100,
        "net_return_pct": net * 100,
        "breakeven_borrow_cost_pct": (gross / horizon_years) * 100,
        "breakeven_value_per_share": price * (1 - carry),
        "max_loss_pct_if_price_doubles": 100.0,
        "reward_to_risk": max(net, 0.0),
        "short_works_at_these_inputs": bool(net > 0),
        "short_interest_shares": 1_780_000,
        "horizon_years": horizon_years,
        "note": ("Gross return is the gap between the observed price and the model value at "
                 "the stated inputs; it is a scenario result, not a recommendation."),
    }


def short_table(base: Optional[ValuationParams] = None,
                horizon_years: float = 1.0) -> pd.DataFrame:
    """Short outcomes under bear, reference and bull inputs."""
    base = base or params_from_assumptions()
    rows = []
    for s in SCENARIOS:
        rows.append({"scenario": s, **short_metrics(scenario_params(base, s), horizon_years)})
    return pd.DataFrame(rows)


def invalidation_conditions(base: Optional[ValuationParams] = None) -> pd.DataFrame:
    """Evidence that would move the model value to or above the price."""
    base = base or params_from_assumptions()
    bundle = required_bundle(base)
    rows = [
        {"category": "Clinical", "condition":
         "Head-to-head or adequately controlled evidence that oral deucrictibant is superior "
         "to injected prophylaxis on attack-free rate at 24+ weeks",
         "why_it_matters": "Justifies a higher peak penetration or price premium.",
         "measured_by": "Randomised trial with a stated window; cross-trial read-across does "
                        "not count.",
         "currently": "Not measured. CHAPTER-3 vs placebo only."},
        {"category": "Clinical", "condition":
         "Durability of the attack-free rate beyond 24 weeks (48+ week extension)",
         "why_it_matters": "Attack-free rates fall with window length; a long window result "
                           "would retire the window-length discount.",
         "measured_by": "Extension study with an explicit measurement window.",
         "currently": "Not measured. 168-day window only."},
        {"category": "Prescribing", "condition":
         "Switching rate above the stated base case in real-world data",
         "why_it_matters": f"Peak penetration is the largest driver of value per share.",
         "measured_by": "Pharmacy-claims based switching cohorts, not stated preferences.",
         "currently": "Not measured. No claims-based switching series in the ledger."},
        {"category": "Pricing", "condition":
         "Net price realised above the assumed level at launch",
         "why_it_matters": "Price moves value per share nearly one-for-one.",
         "measured_by": "Gross-to-net realisation after the first full year of sales.",
         "currently": "Not measured. Pharvaris has not set a price."},
        {"category": "Regulatory", "condition":
         "Approval of both indications with no label restriction",
         "why_it_matters": f"Approval probability is applied once per product; both approvals "
                           f"raise value by the difference between {base.approval_prophylaxis:.0%}"
                           f" / {base.approval_acute:.0%} and 100%.",
         "measured_by": "FDA action (PDUFA 23 Apr 2027 for the on-demand indication).",
         "currently": "Pending. NDA under review."},
        {"category": "Market", "condition":
         "Prevalence or prophylaxis penetration above the ledger's stated values",
         "why_it_matters": "The addressable denominator multiplies directly.",
         "measured_by": "Registry-confirmed diagnosed prevalence, not claims extrapolation.",
         "currently": f"Ledger states 19,614 diagnosed globally; required equity value for "
                      f"${base.stock_price:,.2f} is {bundle['required_equity_value']:,.0f} USD "
                      f"versus a modelled {bundle['stated_equity_value']:,.0f} USD."},
    ]
    return pd.DataFrame(rows)


def explain_approval_vs_adoption() -> str:
    return (
        "**Approval risk and adoption risk answer different questions and must not be "
        "multiplied twice.**\n\n"
        "- *Approval* is binary and regulatory: will the FDA allow the indication? It is "
        "applied **once**, to each product's entire contribution stream.\n"
        "- *Adoption* is continuous and commercial: of the patients who could take it, what "
        "share does? It enters as peak penetration inside the contribution stream, not as a "
        "second probability.\n\n"
        "A model that multiplies an adoption probability by a revenue line that already "
        "contains adoption counts the same uncertainty twice and understates value; a model "
        "that takes the better of the two ignores one of them entirely. Here the separation "
        "is explicit: `contribution_prophylaxis` is the cash flow if approved and adopted at "
        "the stated penetration, and `approval_prophylaxis` scales it once."
    )
