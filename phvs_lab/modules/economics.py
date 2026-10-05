"""Prevention and rescue economics.

Cohorts are **mutually exclusive**: a patient appears in exactly one segment and
exactly one treatment state, so prophylaxis patients and acute-treated patients
are never added together as if they were separate populations.  Acute demand is
derived from the attacks that remain *after* prevention inside the same cohort,
which is where the intentional cannibalisation between PHVS's two products is
captured.

Every figure produced here is either (a) a source fact carried through from the
ledger, or (b) an explicitly labelled model result.  Nothing is a measured
prescribing rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import optimize

from phvs_lab.modules.data_loader import (
    get_assumption,
    get_commercial_assumptions,
)

SEGMENT_PARAMETERS = [
    "seg_well_controlled",
    "seg_incompletely_controlled",
    "seg_injection_averse",
    "seg_poor_daily_pill_adherence",
    "seg_gene_editing_unwilling",
    "seg_residual_no_demand",
]

MODEL_LABEL = "MODEL-DERIVED - NOT A MEASURED PRESCRIBING RATE"


class CohortError(ValueError):
    """Raised when cohorts overlap or do not account for the population."""


# ---------------------------------------------------------------------------
# Population accounting
# ---------------------------------------------------------------------------

def segment_shares(df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Segment shares, validated to be mutually exclusive and exhaustive."""
    df = df if df is not None else get_commercial_assumptions()
    shares = {name: get_assumption(name, df) for name in SEGMENT_PARAMETERS}
    total = float(np.sum(list(shares.values())))
    if any(v < 0 for v in shares.values()):
        raise CohortError(f"Negative segment share in {shares}.")
    if not np.isclose(total, 1.0, atol=1e-6):
        raise CohortError(
            f"Segment shares must sum to 1.0 to keep cohorts mutually exclusive; "
            f"they sum to {total:.6f}."
        )
    return shares


def population_ledger(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Patient accounting from global diagnosed population to addressable cohorts.

    Every stage carries the parameter it came from and whether that parameter is
    a source fact or an analyst assumption.  Stages are non-increasing, and the
    segment split is exhaustive by construction (validated by
    :func:`segment_shares`).
    """
    df = df if df is not None else get_commercial_assumptions()
    shares = segment_shares(df)

    def p(name: str) -> float:
        return get_assumption(name, df)

    us, eu5uk, row = p("us_diagnosed_hae_patients"), p("eu5_uk_diagnosed_hae_patients"), \
        p("row_diagnosed_hae_patients")
    diagnosed = p("global_diagnosed_hae_patients")
    if not np.isclose(us + eu5uk + row, diagnosed, atol=1.0):
        raise CohortError(
            f"Regional populations ({us:.0f} + {eu5uk:.0f} + {row:.0f}) do not sum to the "
            f"global diagnosed population {diagnosed:.0f}."
        )

    eligibility = p("prophylaxis_eligibility_rate")
    coverage = p("payer_coverage_rate")
    penetration = p("prophylaxis_penetration_of_diagnosed")

    eligible = diagnosed * eligibility
    covered = eligible * coverage
    on_ltp = diagnosed * penetration

    rows = [
        {"stage": "Global diagnosed HAE", "patients": diagnosed, "basis": "calculated",
         "parameter": "global_diagnosed_hae_patients",
         "note": "US (verified) + EU5/UK (derived) + RoW (assumption)."},
        {"stage": "Clinically eligible for prophylaxis", "patients": eligible,
         "basis": "assumption", "parameter": "prophylaxis_eligibility_rate",
         "note": "Share of diagnosed patients eligible for long-term prophylaxis."},
        {"stage": "Payer-covered eligible", "patients": covered, "basis": "assumption",
         "parameter": "payer_coverage_rate",
         "note": "Eligible patients whose payer approves a premium oral prophylactic."},
        {"stage": "On long-term prophylaxis today", "patients": on_ltp,
         "basis": "assumption", "parameter": "prophylaxis_penetration_of_diagnosed",
         "note": "Existing LTP share of all diagnosed patients."},
    ]
    ledger = pd.DataFrame(rows)
    if not (ledger["patients"].is_monotonic_decreasing and (ledger["patients"] >= 0).all()):
        raise CohortError(
            "Population stages must be non-increasing: "
            f"{ledger[['stage', 'patients']].to_dict('records')}"
        )

    seg_rows = []
    for name, share in shares.items():
        seg_rows.append({
            "stage": f"Segment: {name.replace('seg_', '').replace('_', ' ')}",
            "patients": eligible * share,
            "basis": "assumption", "parameter": name,
            "note": f"{share:.0%} of the prophylaxis-eligible population.",
        })
    segments = pd.DataFrame(seg_rows)
    if not np.isclose(segments["patients"].sum(), eligible, atol=1e-6):
        raise CohortError(
            f"Segment cohorts sum to {segments['patients'].sum():.2f} but the eligible "
            f"population is {eligible:.2f}; patients would be double counted or lost."
        )
    return pd.concat([ledger, segments], ignore_index=True)


# ---------------------------------------------------------------------------
# Rate / dispersion calibration
# ---------------------------------------------------------------------------

def fit_dispersion(mean_window_attacks: float, observed_af: float) -> float:
    """Patient-level dispersion that reproduces an observed attack-free rate.

    Under a gamma-mixed Poisson process the probability of zero attacks in a
    window with mean ``mu`` is ``(k/(k+mu))**k``.  Given a published mean rate
    and a published attack-free proportion for the same window, this solves for
    ``k``.  ``np.inf`` is returned when the observed rate is at or above the
    homogeneous Poisson value (no heterogeneity needed).

    This is a calibration step, not clinical evidence.
    """
    if not 0 < observed_af < 1:
        raise ValueError(f"observed_af must be in (0, 1); got {observed_af}.")
    if mean_window_attacks <= 0:
        return float(np.inf)
    poisson_af = float(np.exp(-mean_window_attacks))
    if observed_af <= poisson_af:
        return float(np.inf)

    def objective(log_k: float) -> float:
        k = float(np.exp(log_k))
        return float((k / (k + mean_window_attacks)) ** k - observed_af)

    log_k = optimize.brentq(objective, np.log(1e-3), np.log(1e6), xtol=1e-12, rtol=1e-10)
    return float(np.exp(log_k))


def attack_free_probability(mean_window_attacks: float, dispersion: float = np.inf) -> float:
    if mean_window_attacks <= 0:
        return 1.0
    if np.isinf(dispersion):
        return float(np.exp(-mean_window_attacks))
    return float((dispersion / (dispersion + mean_window_attacks)) ** dispersion)


# ---------------------------------------------------------------------------
# Prevention -> rescue bridge
# ---------------------------------------------------------------------------

@dataclass
class PreventionRescueParams:
    """Editable inputs for the prevention / rescue bridge.

    ``trial_efficacy`` / ``trial_window_days`` / ``observed_attack_free`` are the
    published values used once to calibrate patient-level dispersion.  They stay
    fixed while ``efficacy`` is varied, so a scenario change moves the answer for
    the expected reason rather than by re-fitting the calibration.
    """
    cohort_patients: float
    baseline_rate_per_4wk: float = 1.93     # CHAPTER-1 placebo LS-mean (source fact)
    efficacy: float = 0.83                  # scenario efficacy
    trial_efficacy: float = 0.83            # CHAPTER-3 vs placebo (calibration input)
    trial_window_days: float = 168.0        # CHAPTER-3 window (calibration input)
    observed_attack_free: float = 0.4545    # CHAPTER-3 active arm (calibration input)
    real_world_haircut: float = 0.75        # analyst assumption
    window_days: float = 168.0              # window used for the projection
    prophylaxis_price: float = 425_000.0    # assumed annual net price
    acute_price: float = 85_000.0           # assumed annual spend per acute-treated patient
    acute_share_of_breakthrough: float = 1.0  # share of breakthrough patients buying acute
    dispersion: Optional[float] = None      # override; None => calibrate from the trial


def calibrate_dispersion(params: PreventionRescueParams) -> float:
    """Fit the heterogeneity parameter once, from the trial's own published values."""
    trial_months = params.trial_window_days / 30.4375
    trial_mean = params.baseline_rate_per_4wk * (1 - params.trial_efficacy) * trial_months
    return fit_dispersion(trial_mean, params.observed_attack_free)


def bridge_components(params: PreventionRescueParams) -> Dict[str, float]:
    """Prevention and rescue economics for one cohort, with no patient reuse.

    The cohort size appears once.  Prophylaxis revenue is earned by the whole
    cohort; acute revenue is earned only by the subset that still breaks through,
    so prevention can only reduce acute revenue (the intentional cannibalisation
    between PHVS's two products).
    """
    if params.cohort_patients < 0:
        raise CohortError("cohort_patients must be non-negative.")
    if not 0 <= params.efficacy <= 1 or not 0 <= params.real_world_haircut <= 1:
        raise ValueError("efficacy and real_world_haircut must be proportions in [0, 1].")

    k = params.dispersion if params.dispersion is not None else calibrate_dispersion(params)
    effective_efficacy = params.efficacy * params.real_world_haircut
    residual_rate_4wk = params.baseline_rate_per_4wk * (1 - effective_efficacy)
    months = params.window_days / 30.4375
    af_model = attack_free_probability(residual_rate_4wk * months, k)
    breakthrough_patients = (params.cohort_patients * (1 - af_model)
                             * params.acute_share_of_breakthrough)

    proph_revenue = params.cohort_patients * params.prophylaxis_price
    acute_revenue = breakthrough_patients * params.acute_price

    periods_per_year = 365.25 / 28.0        # four-week periods in a year
    untreated_attacks = (params.cohort_patients * params.baseline_rate_per_4wk
                         * periods_per_year)
    treated_attacks = params.cohort_patients * residual_rate_4wk * periods_per_year

    return {
        "patients": params.cohort_patients,
        "effective_efficacy": effective_efficacy,
        "residual_rate_per_4wk": residual_rate_4wk,
        "fitted_dispersion_k": k,
        "model_attack_free": af_model,
        "breakthrough_patients": breakthrough_patients,
        "untreated_attacks_per_year": untreated_attacks,
        "on_treatment_attacks_per_year": treated_attacks,
        "attacks_averted_per_year": untreated_attacks - treated_attacks,
        "prophylaxis_revenue": proph_revenue,
        "acute_revenue": acute_revenue,
        "total_revenue": proph_revenue + acute_revenue,
        "calibration_note": (
            f"Dispersion k={k:.4g} calibrated once so the model reproduces "
            f"{params.observed_attack_free:.1%} attack-free over "
            f"{params.trial_window_days:.0f} days at the trial's reported "
            f"{params.trial_efficacy:.0%} rate reduction; held fixed while the scenario "
            f"efficacy changes."
        ) if np.isfinite(k) else (
            "Observed attack-free rate is at or below the homogeneous Poisson value, so no "
            "patient-level heterogeneity is required to reproduce it."
        ),
        "label": MODEL_LABEL,
    }


def efficacy_projection(params: PreventionRescueParams,
                        efficacies: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Acute demand and revenue as prevention efficacy varies (single calibration)."""
    efficacies = np.linspace(0.40, 1.00, 25) if efficacies is None else np.asarray(efficacies)
    rows = []
    for eff in efficacies:
        p = PreventionRescueParams(**{**params.__dict__, "efficacy": float(eff)})
        c = bridge_components(p)
        rows.append({"efficacy": float(eff),
                     "effective_efficacy": c["effective_efficacy"],
                     "model_attack_free": c["model_attack_free"],
                     "breakthrough_patients": c["breakthrough_patients"],
                     "acute_revenue": c["acute_revenue"],
                     "prophylaxis_revenue": c["prophylaxis_revenue"],
                     "total_revenue": c["total_revenue"],
                     "label": MODEL_LABEL})
    return pd.DataFrame(rows)


def prevention_rescue_ladder(params: PreventionRescueParams) -> pd.DataFrame:
    """Stage-by-stage ladder: patients -> attacks averted -> prescriptions -> revenue."""
    c = bridge_components(params)
    rows = [
        ("Cohort patients (counted once)", c["patients"], "patients", "exposure"),
        ("Untreated attacks per year", c["untreated_attacks_per_year"], "attacks",
         "clinical event - not a prescription"),
        ("On-treatment attacks per year", c["on_treatment_attacks_per_year"], "attacks",
         "residual events after prevention"),
        ("Attacks averted per year", c["attacks_averted_per_year"], "attacks",
         "difference of the two rows above"),
        ("Breakthrough patients (buy acute)", c["breakthrough_patients"], "patients",
         "subset of the cohort, never added to it"),
        ("Prophylaxis revenue", c["prophylaxis_revenue"], "USD", "whole cohort x net price"),
        ("Acute revenue", c["acute_revenue"], "USD", "breakthrough subset x annual spend"),
        ("Total revenue", c["total_revenue"], "USD", "sum of the two revenue lines"),
    ]
    out = pd.DataFrame(rows, columns=["stage", "value", "unit", "basis"])
    out["status"] = MODEL_LABEL
    return out


def assert_no_double_counting(ledger: pd.DataFrame, cohort_sizes: Dict[str, float],
                              total_population: float, tol: float = 1e-6) -> None:
    """Raise if any patient would be counted in two cohorts or beyond the population."""
    total = float(sum(cohort_sizes.values()))
    if total > total_population + tol:
        raise CohortError(
            f"Cohorts total {total:.4f} patients, more than the population "
            f"{total_population:.4f}; patients are being double counted."
        )
    if any(v < 0 for v in cohort_sizes.values()):
        raise CohortError("Negative cohort size: cohorts must be non-negative.")


def explain_double_counting() -> str:
    return (
        "**Where the double-counting error happens.**\n\n"
        "The tempting arithmetic is *prophylaxis patients x price + acute patients x price*. "
        "Most prophylaxis patients are also acute-treated patients for breakthrough attacks, so "
        "adding the two populations counts the same person twice and inflates the addressable "
        "market.\n\n"
        "This lab instead:\n"
        "1. splits the prophylaxis-eligible population into **mutually exclusive segments** "
        "(validated to sum to 1.0);\n"
        "2. counts every cohort patient **once** for prophylaxis revenue;\n"
        "3. derives acute demand from the **breakthrough subset inside that same cohort**, so "
        "better prevention mechanically lowers acute revenue;\n"
        "4. keeps *attacks*, *treated attacks*, *patients who buy acute therapy* and *revenue* "
        "as four separate quantities, each with its own conversion assumption.\n\n"
        "Result: prevention cannibalising rescue revenue is visible as a number instead of "
        "being hidden by two overlapping patient counts."
    )


def summarize_economics(df: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Ledger + bridge at the default (reference) inputs."""
    df = df if df is not None else get_commercial_assumptions()
    ledger = population_ledger(df)
    eligible = ledger.loc[ledger["stage"] == "Clinically eligible for prophylaxis",
                          "patients"].iloc[0]
    shares = segment_shares(df)
    params = PreventionRescueParams(
        cohort_patients=eligible * shares["seg_incompletely_controlled"],
        efficacy=get_assumption("chapter3_rate_reduction_vs_placebo", df),
        real_world_haircut=get_assumption("real_world_efficacy_haircut", df),
        baseline_rate_per_4wk=get_assumption("baseline_monthly_attack_rate", df),
        prophylaxis_price=get_assumption("phvs_prophylaxis_annual_net_price_us", df),
        acute_price=get_assumption("phvs_acute_annual_net_price_us", df),
    )
    cohort_sizes = {k: eligible * v for k, v in shares.items()}
    assert_no_double_counting(ledger, cohort_sizes, eligible)
    return {"ledger": ledger, "params": params, "bridge": bridge_components(params),
            "ladder": prevention_rescue_ladder(params), "cohorts": cohort_sizes}
