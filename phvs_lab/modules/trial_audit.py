"""Trial comparability audit.

Places every result next to the population, window, endpoint, background
therapy and source that produced it, and returns explicit flags whenever two
results are being read side by side.  No two separate trials are ever presented
as a randomised comparison; window standardisation is offered only as a
labelled model calculation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from phvs_lab.modules.data_loader import endpoint_family, get_clinical_trials
from phvs_lab.modules.response_engine import attack_free_probability

DISPLAY_RENAME = {
    "drug": "Drug", "trial": "Trial", "treatment_arm": "Arm",
    "population": "Population", "sample_size": "N",
    "endpoint_definition": "Endpoint", "measurement_window": "Window",
    "numerator": "Num.", "denominator": "Den.", "proportion": "Result",
    "background_treatment": "Background therapy",
    "value_type": "Value type", "verification_status": "Verification",
    "source_url": "Source",
}

ILLUSTRATIVE = "MODEL-DERIVED - NOT ADJUSTED CLINICAL EVIDENCE"


# ---------------------------------------------------------------------------
# Comparability table
# ---------------------------------------------------------------------------

def build_comparability_table(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every observation with the facts that determine whether it is comparable."""
    df = df if df is not None else get_clinical_trials()
    cols = [c for c in DISPLAY_RENAME if c in df.columns]
    out = df[cols].rename(columns=DISPLAY_RENAME).copy()
    out["Result"] = df["proportion"]
    out["Result"] = out["Result"].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    out = out.sort_values(["Endpoint", "Window", "Drug", "Trial"])
    return out.reset_index(drop=True)


def trial_design(trial: str, df: Optional[pd.DataFrame] = None) -> str:
    """Design label inferred from the recorded window / trial name."""
    df = df if df is not None else get_clinical_trials()
    sub = df[df["trial"] == trial]
    text = " ".join(sub["measurement_window"].dropna().astype(str)).lower()
    if "crossover" in text or "cross-over" in text:
        return "two-period crossover"
    if "open" in text:
        return "open-label"
    if sub["treatment_arm"].str.contains("Placebo", case=False, na=False).any():
        return "randomised, placebo-controlled"
    return "controlled (design not recorded)"


def comparator_trials(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per (trial, arm) with the design facts needed for side-by-side reading."""
    df = df if df is not None else get_clinical_trials()
    rows = []
    for (drug, trial, arm), g in df.groupby(["drug", "trial", "treatment_arm"]):
        rows.append({
            "Drug": drug, "Trial": trial, "Arm": arm,
            "Population": g["population"].dropna().iloc[0] if g["population"].notna().any() else "",
            "N": g["sample_size"].max(),
            "Design": trial_design(trial),
            "Window": "; ".join(sorted(set(g["measurement_window"].dropna()))),
            "Window days": g["measurement_window_days"].max(),
            "Endpoints": g["endpoint_definition"].nunique(),
            "Background therapy": g["background_treatment"].dropna().iloc[0]
                if g["background_treatment"].notna().any() else "",
            "Verification": "; ".join(sorted(set(g["verification_status"]))),
            "Source": g["source_id"].dropna().iloc[0] if g["source_id"].notna().any() else "",
        })
    return pd.DataFrame(rows).sort_values(["Trial", "Arm"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pairwise flags
# ---------------------------------------------------------------------------

def _flag(severity: str, code: str, message: str) -> Dict:
    return {"severity": severity, "code": code, "message": message}


def compare_arms(df: Optional[pd.DataFrame] = None, trial_a: str = "", trial_b: str = ""
                 ) -> List[Dict]:
    """Explicit flags for reading two trial arms together.

    The first flag is always raised: separate trials are not a randomised
    comparison, regardless of how similar the numbers look.
    """
    df = df if df is not None else get_clinical_trials()
    flags: List[Dict] = [_flag(
        "blocking", "not_randomised",
        f"'{trial_a}' and '{trial_b}' are separate studies. Reading them together is a "
        f"cross-trial comparison, never a randomised head-to-head result."
    )]
    if not trial_a or not trial_b or trial_a == trial_b:
        return flags

    a = df[df["trial"] == trial_a]
    b = df[df["trial"] == trial_b]
    if a.empty or b.empty:
        flags.append(_flag("blocking", "unknown_trial", "One of the selected trials has no recorded observations."))
        return flags

    da, db = trial_design(trial_a, a), trial_design(trial_b, b)
    if da != db:
        flags.append(_flag("blocking", "different_design",
                           f"Design differs: {da} vs {db}."))

    pop_a = set(a["population"].dropna())
    pop_b = set(b["population"].dropna())
    if pop_a != pop_b:
        flags.append(_flag("blocking", "different_population",
                           f"Populations differ: {sorted(pop_a)} vs {sorted(pop_b)}."))

    win_a = a["measurement_window_days"].max()
    win_b = b["measurement_window_days"].max()
    if not (np.isnan(win_a) or np.isnan(win_b)) and win_a != win_b:
        ratio = max(win_a, win_b) / min(win_a, win_b)
        flags.append(_flag(
            "blocking" if ratio >= 1.5 else "material", "different_window",
            f"Observation windows differ ({win_a:.0f} vs {win_b:.0f} days, {ratio:.2f}x). "
            f"Attack-free rates fall mechanically with window length."))

    endpoints_a = set(a["endpoint_definition"].dropna())
    endpoints_b = set(b["endpoint_definition"].dropna())
    if endpoints_a != endpoints_b:
        flags.append(_flag("material", "different_endpoint",
                           "Endpoint definitions are not identical; check the exact wording "
                           "and the denominator used."))

    bg_a = set(a["background_treatment"].dropna())
    bg_b = set(b["background_treatment"].dropna())
    if bg_a != bg_b:
        flags.append(_flag("material", "different_background",
                           f"Background therapy differs: {sorted(bg_a) or ['none recorded']} vs "
                           f"{sorted(bg_b) or ['none recorded']}."))

    n_a, n_b = a["sample_size"].max(), b["sample_size"].max()
    if min(n_a, n_b) < 30:
        flags.append(_flag("material", "small_sample",
                           f"Arm sizes {n_a:.0f} and {n_b:.0f}: one patient moves the rate by "
                           f"{100/min(n_a, n_b):.1f} pp."))

    status_a = "; ".join(sorted(set(a["verification_status"])))
    status_b = "; ".join(sorted(set(b["verification_status"])))
    if "provisional" in status_a or "provisional" in status_b:
        flags.append(_flag("blocking", "unverified_value",
                           f"Unverified values in scope ({status_a} / {status_b}). "
                           f"Comparison cannot be relied upon until re-verified."))
    return flags


def comparability_warnings(df: Optional[pd.DataFrame] = None) -> List[Dict]:
    """Warnings where the same endpoint family is defined or measured differently."""
    df = df if df is not None else get_clinical_trials()
    out: List[Dict] = []
    df = df.copy()
    df["family"] = df["endpoint_definition"].map(endpoint_family)
    for family, g in df.groupby("family"):
        trials = sorted(set(g["trial"]))
        windows = sorted(set(g["measurement_window"].dropna()))
        populations = sorted(set(g["population"].dropna()))
        if len(trials) > 1 and len(windows) > 1:
            out.append(_flag(
                "blocking", "family_window_mismatch",
                f"Endpoint family '{family}' appears in {', '.join(trials)} with windows "
                f"{windows}. Longer windows mechanically lower responder and attack-free "
                f"rates, so these values are not directly comparable."))
        elif len(trials) > 1:
            out.append(_flag(
                "material", "family_across_trials",
                f"Endpoint family '{family}' appears in {', '.join(trials)}. Separate trials; "
                f"pooling requires meta-analysis, not averaging."))
        if len(populations) > 1:
            out.append(_flag(
                "material", "family_population_mismatch",
                f"Endpoint family '{family}' is studied in different populations: "
                f"{populations}."))
    return out


# ---------------------------------------------------------------------------
# Window mechanics
# ---------------------------------------------------------------------------

def explain_window_mechanics() -> str:
    return (
        "**Why window length is not a detail.**\n\n"
        "If attacks arrive at a constant monthly rate $\\lambda$, the probability of "
        "observing zero attacks over a window of $T$ days is $e^{-\\lambda T}$. Doubling the "
        "window squares the attack-free probability's complement on the log scale: a rate that "
        "gives 50% attack-free in 4 weeks gives about 3% in 24 weeks with no change in the "
        "underlying therapy.\n\n"
        "Overdispersion softens but does not remove the effect: with patient-level heterogeneity "
        "(gamma-Poisson, dispersion $k$) the probability is $(k/(k+\\mu))^{k}$, which stays above "
        "the Poisson value but still falls monotonically in the window.\n\n"
        "Therefore an attack-free rate from a 12-week study, a 24-week study and a 26-week study "
        "are three different quantities. They may be displayed together only with their windows "
        "and with the standardised values flagged as model-derived."
    )


def standardise_attack_free(observed: float, window_days: float,
                            target_days: float, dispersion: float = np.inf) -> float:
    """Translate an observed attack-free rate to a different window length.

    Implied rate under the chosen count model: ``lambda = -ln(p) / window`` for
    Poisson, or the gamma-mixed equivalent, then re-evaluated at ``target_days``.

    Returns a model-derived figure, never a measured one.
    """
    if not 0 < observed < 1:
        return float("nan")
    if np.isinf(dispersion):
        lam = -np.log(observed) / window_days
        return float(np.exp(-lam * target_days))
    # gamma-Poisson: p = (k/(k+mu))^k  =>  mu = k * (p^(-1/k) - 1)
    mu = dispersion * (observed ** (-1.0 / dispersion) - 1.0)
    lam = mu / window_days
    return float((dispersion / (dispersion + lam * target_days)) ** dispersion)


def window_adjusted_table(df: Optional[pd.DataFrame] = None,
                          target_days: float = 168.0,
                          dispersion: float = np.inf) -> pd.DataFrame:
    """Attack-free results re-expressed on a common window (model-derived).

    Only arms with a published attack-free count are included.  Every row is
    labelled ``ILLUSTRATIVE``: the standardisation assumes a constant rate and
    is not an adjusted clinical comparison.
    """
    df = df if df is not None else get_clinical_trials()
    rows = []
    for _, r in df[df["endpoint_definition"].str.contains("attack-free", case=False, na=False)
                   & df["is_count"]].iterrows():
        p = r["proportion"]
        std = standardise_attack_free(p, float(r["measurement_window_days"]),
                                      target_days, dispersion)
        rows.append({
            "Drug": r["drug"], "Trial": r["trial"], "Arm": r["treatment_arm"],
            "N": int(r["denominator"]),
            "Window days": int(r["measurement_window_days"]),
            "Observed attack-free": p,
            f"Standardised to {int(target_days)}d": std,
            "Verification": r["verification_status"],
            "Status": ILLUSTRATIVE,
            "Source": r["source_url"],
        })
    return pd.DataFrame(rows)


def window_sensitivity_curves(monthly_rates: Sequence[float],
                              dispersions: Sequence[float] = (np.inf, 5.0, 1.5),
                              windows_days: Sequence[float] = (28, 84, 168, 182, 365)
                              ) -> pd.DataFrame:
    """Analytic attack-free probability surface (illustrative)."""
    rows = []
    for rate in monthly_rates:
        for disp in dispersions:
            for w in windows_days:
                rows.append({
                    "monthly_rate": rate,
                    "rate_model": "Poisson" if np.isinf(disp) else f"gamma-mixed (k={disp:g})",
                    "dispersion": disp,
                    "window_days": w,
                    "attack_free_probability": attack_free_probability(rate, disp, w),
                    "Status": ILLUSTRATIVE,
                })
    return pd.DataFrame(rows)


def build_comparability_warnings(trials_df: pd.DataFrame) -> List[Dict]:
    """Backwards-compatible alias."""
    return comparability_warnings(trials_df)
