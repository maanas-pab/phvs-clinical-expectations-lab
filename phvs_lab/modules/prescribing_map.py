"""Prescribing map: segments, choice model and the price break-even.

The map answers one question: **given stated preferences, what prescribing is
required for the observed price?**

* Segments are mutually exclusive and validated to sum to 1 (shared with the
  economics ledger).
* Choice shares come from a multinomial logit over three options, with every
  coefficient carried in the assumption ledger rather than hidden in code.
* The resulting patient count is converted into peak penetration and fed into
  the valuation, so the contour is in *value per share* - the same units as the
  market price.

Nothing here is a measured prescribing probability. The contours are scenario
outputs of the stated inputs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from phvs_lab.modules.data_loader import (
    get_assumption,
    get_commercial_assumptions,
)

__all__ = [
    "SEGMENTS", "MapParams", "choice_probabilities", "implied_patients",
    "implied_peak_penetration", "value_for_inputs", "value_surface",
    "break_even_switching", "segment_break_even", "explain_choice_model",
]

MODEL_LABEL = "SCENARIO OUTPUT - NOT A MEASURED PRESCRIBING RATE"

# (label, share parameter, oral-preference parameter, addressability parameter)
SEGMENTS: List[Tuple[str, str, str, str]] = [
    ("Well controlled on current prophylaxis", "seg_well_controlled",
     "oral_pref_well_controlled", "addressable_well_controlled"),
    ("Incompletely controlled", "seg_incompletely_controlled",
     "oral_pref_incompletely_controlled", "addressable_incompletely_controlled"),
    ("Injection averse", "seg_injection_averse",
     "oral_pref_injection_averse", "addressable_injection_averse"),
    ("Poor daily-pill adherence", "seg_poor_daily_pill_adherence",
     "oral_pref_poor_daily_pill_adherence", "addressable_poor_daily_pill_adherence"),
    ("Unwilling or ineligible for gene editing", "seg_gene_editing_unwilling",
     "oral_pref_gene_editing_unwilling", "addressable_gene_editing_unwilling"),
    ("Residual: no demand", "seg_residual_no_demand",
     "oral_pref_residual_no_demand", "addressable_residual_no_demand"),
]

OPTIONS = ("stay on current prophylaxis", "switch to existing oral",
           "switch to PHVS oral")


class MapError(ValueError):
    """Raised for infeasible prescribing-map inputs."""


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class MapParams:
    """Choice-model parameters read from the assumption ledger."""

    def __init__(self, df: Optional[pd.DataFrame] = None,
                 premium_multiplier: float = 1.0,
                 switching_rate: Optional[float] = None):
        self.df = df if df is not None else get_commercial_assumptions()
        self.premium_multiplier = float(premium_multiplier)
        self.switching_rate = float(
            get_assumption("base_switching_rate", self.df)
            if switching_rate is None else switching_rate)

        def a(name: str) -> float:
            return get_assumption(name, self.df)

        self.ref_price = a("reference_ltp_price_usd")
        self.stated_price = a("phvs_prophylaxis_annual_net_price_us")
        self.market_anchor = a("implied_revenue_per_takhyro_patient")
        self.existing_oral_index = a("existing_oral_price_index")
        self.asc_current = a("mnl_asc_current")
        self.asc_existing_oral = a("mnl_asc_existing_oral")
        self.asc_phvs_oral = a("mnl_asc_phvs_oral")
        self.price_coef = a("mnl_price_coef_per_ref")
        self.switch_cost = a("mnl_switch_cost")
        self.efficacy_bonus = a("mnl_efficacy_bonus")
        self.discontinuation = a("prophylaxis_annual_discontinuation")
        self.time_to_peak = a("prophylaxis_time_to_peak_years")
        self.eligible = a("prophylaxis_eligible_patients_global")

        self.shares = {label: a(share_p) for label, share_p, _, _ in SEGMENTS}
        self.oral_pref = {label: a(pref_p) for label, _, pref_p, _ in SEGMENTS}
        self.addressable = {label: a(addr_p) for label, _, _, addr_p in SEGMENTS}
        self._validate()

    def _validate(self) -> None:
        total = float(sum(self.shares.values()))
        if not np.isclose(total, 1.0, atol=1e-6):
            raise MapError(f"Segment shares must sum to 1.0; they sum to {total:.6f}.")
        if any(not 0 <= v <= 1 for v in self.addressable.values()):
            raise MapError(f"Addressability must lie in [0, 1]: {self.addressable}")
        if not 0 <= self.switching_rate <= 1:
            raise MapError(f"switching_rate must lie in [0, 1]; got {self.switching_rate}.")
        if self.premium_multiplier <= 0:
            raise MapError(f"premium_multiplier must be positive; got {self.premium_multiplier}.")
        if not 0 <= self.discontinuation <= 1:
            raise MapError("discontinuation must lie in [0, 1].")

    @property
    def phvs_price(self) -> float:
        return self.stated_price * self.premium_multiplier

    @property
    def cumulative_switch(self) -> float:
        """Share of an addressable patient pool that has moved by the peak year."""
        return 1.0 - (1.0 - self.switching_rate) ** self.time_to_peak


# ---------------------------------------------------------------------------
# Choice model
# ---------------------------------------------------------------------------

def _softmax(u: np.ndarray, axis: int = -1) -> np.ndarray:
    z = u - np.max(u, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def choice_probabilities(params: Optional[MapParams] = None) -> pd.DataFrame:
    """Multinomial-logit choice shares for each segment at the stated price."""
    p = params if params is not None else MapParams()

    rows = []
    for label, _, _, _ in SEGMENTS:
        pref = p.oral_pref[label]
        price_current = p.ref_price / p.ref_price                 # = 1 reference unit
        price_existing = p.existing_oral_index
        price_phvs = p.phvs_price / p.ref_price
        u = np.array([
            p.asc_current + p.price_coef * price_current,
            p.asc_existing_oral + pref + p.price_coef * price_existing + p.switch_cost,
            p.asc_phvs_oral + pref + p.price_coef * price_phvs + p.switch_cost
            + p.efficacy_bonus,
        ])
        probs = _softmax(u)
        rows.append({
            "segment": label,
            "share_of_population": p.shares[label],
            "addressable": p.addressable[label],
            "oral_preference": pref,
            "utility_current": u[0],
            "utility_existing_oral": u[1],
            "utility_phvs_oral": u[2],
            "p_current": probs[0],
            "p_existing_oral": probs[1],
            "p_phvs_oral": probs[2],
            "label": MODEL_LABEL,
        })
    return pd.DataFrame(rows)


def implied_patients(params: Optional[MapParams] = None) -> pd.DataFrame:
    """Segment-by-segment patient build, with the funnel made explicit."""
    p = params if params is not None else MapParams()
    choices = choice_probabilities(p)
    cum = p.cumulative_switch
    persistence = 1.0 - p.discontinuation

    out = choices.copy()
    out["eligible_in_segment"] = p.eligible * out["share_of_population"]
    out["addressable_patients"] = out["eligible_in_segment"] * out["addressable"]
    out["switchers"] = out["addressable_patients"] * cum
    out["phvs_patients"] = out["switchers"] * out["p_phvs_oral"] * persistence
    out["cumulative_switch"] = cum
    out["persistence_multiplier"] = persistence
    return out


def implied_peak_penetration(params: Optional[MapParams] = None) -> float:
    """Peak prophylaxis penetration implied by the map (patients / eligible)."""
    p = params if params is not None else MapParams()
    patients = implied_patients(p)["phvs_patients"].sum()
    return float(patients / p.eligible)


# ---------------------------------------------------------------------------
# Link to value
# ---------------------------------------------------------------------------

def value_for_inputs(premium_multiplier: float, switching_rate: float,
                     base=None) -> Dict[str, float]:
    """Value per share implied by a (premium, switching) pair."""
    from phvs_lab.modules import valuation as val

    base = base if base is not None else val.params_from_assumptions()
    mp = MapParams(premium_multiplier=premium_multiplier, switching_rate=switching_rate)
    penetration = implied_peak_penetration(mp)
    vp = replace(base,
                 peak_penetration_prophylaxis=min(penetration, 1.0),
                 price_prophylaxis=mp.phvs_price)
    summary = val.value_summary(vp)
    return {"premium_multiplier": premium_multiplier, "switching_rate": switching_rate,
            "phvs_price": mp.phvs_price, "implied_patients": implied_patients(mp)["phvs_patients"].sum(),
            "implied_peak_penetration": penetration, **summary}


def value_surface(premiums: Optional[Sequence[float]] = None,
                  switchings: Optional[Sequence[float]] = None,
                  base=None) -> pd.DataFrame:
    """Value per share over a (premium x switching) grid - the contour input."""
    premiums = np.linspace(0.6, 1.5, 19) if premiums is None else np.asarray(premiums, float)
    switchings = np.linspace(0.02, 0.50, 19) if switchings is None else np.asarray(switchings, float)
    rows = []
    for sw in switchings:
        for pr in premiums:
            rows.append(value_for_inputs(pr, sw, base))
    out = pd.DataFrame(rows)
    out.attrs["premiums"] = premiums
    out.attrs["switchings"] = switchings
    out.attrs["label"] = MODEL_LABEL
    return out


def break_even_switching(premium_multiplier: float, target_price: Optional[float] = None,
                         base=None) -> Dict[str, float]:
    """Switching rate required for the model to equal the market price."""
    from phvs_lab.modules import valuation as val

    base = base if base is not None else val.params_from_assumptions()
    target = base.stock_price if target_price is None else target_price

    def f(sw: float) -> float:
        return value_for_inputs(premium_multiplier, sw, base)["value_per_share"] - target

    lo, hi = 1e-4, 0.95
    f_lo, f_hi = f(lo), f(hi)
    result = {"premium_multiplier": premium_multiplier, "target_price": target,
              "required_switching_rate": np.nan, "attainable": False,
              "value_at_stated_switching": value_for_inputs(
                  premium_multiplier, get_assumption("base_switching_rate"), base
              )["value_per_share"],
              "value_at_zero_switching": value_for_inputs(premium_multiplier, lo, base)["value_per_share"],
              "value_at_max_switching": value_for_inputs(premium_multiplier, hi, base)["value_per_share"],
              "note": ""}
    if f_lo <= 0 <= f_hi:
        root = optimize.brentq(f, lo, hi, xtol=1e-8, rtol=1e-6)
        result.update(required_switching_rate=float(root), attainable=True,
                      note="Value equals the price at this switching rate, all else equal.")
    elif f_lo > 0 and f_hi > 0:
        result["note"] = (f"The model already exceeds ${target:,.2f} at a 0.01% switching "
                          f"rate; a lower switching rate - or a lower price - is required.")
    else:
        result["note"] = (f"Even at a 95% switching rate the model is worth "
                          f"{value_for_inputs(premium_multiplier, hi, base)['value_per_share']:,.2f}, "
                          f"below ${target:,.2f}. Price or preference must move.")
    return result


def segment_break_even(base=None, target_price: Optional[float] = None
                       ) -> pd.DataFrame:
    """Switching rate required in each segment alone, others held at the stated rate."""
    from phvs_lab.modules import valuation as val

    base = base if base is not None else val.params_from_assumptions()
    target = base.stock_price if target_price is None else target_price
    stated = get_assumption("base_switching_rate")

    rows = []
    for label, _, _, _ in SEGMENTS:
        def value(sw: float, segment: str = label) -> float:
            mp = MapParams(switching_rate=sw)
            tab = implied_patients(mp)
            tab.loc[tab["segment"] != segment, "phvs_patients"] = 0.0
            pen = float(tab["phvs_patients"].sum() / mp.eligible)
            vp = replace(base, peak_penetration_prophylaxis=min(pen, 1.0))
            return val.value_summary(vp)["value_per_share"]

        f_lo, f_hi = value(1e-4) - target, value(0.95) - target
        row = {"segment": label, "target_price": target, "required_switching_rate": np.nan,
               "attainable": False, "patients_if_sole_driver": np.nan,
               "value_at_stated_switching": round(value(stated), 2),
               "max_value_attainable": round(value(0.95), 2), "note": ""}
        if f_lo <= 0 <= f_hi:
            root = optimize.brentq(lambda sw: value(sw) - target, 1e-4, 0.95,
                                   xtol=1e-8, rtol=1e-6)
            mp = MapParams(switching_rate=root)
            tab = implied_patients(mp)
            tab.loc[tab["segment"] != label, "phvs_patients"] = 0.0
            row.update(required_switching_rate=float(root), attainable=True,
                       patients_if_sole_driver=float(tab["phvs_patients"].sum()),
                       note=f"Switching in this segment alone (others at {stated:.0%}) "
                            f"reproduces the price.")
        else:
            row["note"] = (f"This segment alone cannot carry the price: even at a 95% "
                           f"switching rate it is worth ${value(0.95):,.2f} against "
                           f"${target:,.2f}.")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def explain_choice_model() -> str:
    return (
        "**What the map assumes, in order.**\n\n"
        "1. The prophylaxis-eligible population is split into **six mutually exclusive "
        "segments** whose shares are validated to sum to 1, so no patient appears twice.\n"
        "2. Each segment has an **addressability** (share that might change therapy at all) "
        "and an **oral preference** increment.\n"
        "3. Choice among *stay on current prophylaxis*, *switch to the existing oral* and "
        "*switch to PHVS oral* is a **multinomial logit** over utilities that include "
        "intercepts, an oral preference, a switch cost and a price term scaled by annual net "
        "cost. Every coefficient is an assumption in the ledger, editable and visible.\n"
        "4. Switchers accumulate at the stated annual switching rate over the time to peak, "
        "and are reduced by the stated discontinuation rate.\n"
        "5. The resulting patients are divided by the eligible population to give **peak "
        "penetration**, which is passed to the valuation - so the contour reads in value per "
        "share, the same unit as the market price.\n\n"
        "Stated-preference choice shares are **not** observed prescribing behaviour. That is "
        "precisely the gap the survey section is designed to measure - and the gap that "
        "remains unmeasured until real switching data exists.\n\n"
        "**Two edges of the surface.** A switching rate of zero means the programme is funded "
        "and nobody switches, so pre-launch spend with no revenue drives value below zero; it "
        "is a mathematical bound, not a business scenario. A switching rate of 95% is the "
        "opposite bound. The contour between them is where the argument lives."
    )
