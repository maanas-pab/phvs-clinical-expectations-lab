"""Response distribution engine.

Reconstructs **mutually exclusive** response categories from nested, cumulative
thresholds (>=50% / >=70% / >=90% / attack-free) and, when only some thresholds
were disclosed, reports the *identified set* rather than a point estimate.

Nothing in this module invents patient-level data. Where a statistic is a
sampling quantity it is reported as an exact binomial interval; where it is a
design quantity (observation window, sample size) it is reported separately so
the two are never conflated.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from phvs_lab.modules.data_loader import get_clinical_trials

# Cumulative threshold levels. Higher level = stricter responder definition.
LEVELS = {1: ">=50% reduction", 2: ">=70% reduction", 3: ">=90% reduction",
          4: "attack-free"}
CATEGORY_ORDER = [
    "attack-free (100%)",
    "90-99% reduction",
    "70-89% reduction",
    "50-69% reduction",
    "<50% reduction (non-responder)",
]


# ---------------------------------------------------------------------------
# Point reconstruction
# ---------------------------------------------------------------------------

@dataclass
class ResponseDistribution:
    """Mutually exclusive response categories reconstructed from nested thresholds."""

    trial: str
    arm: str
    total_patients: int
    cumulative: Dict[int, int]           # level -> cumulative count
    observed_levels: List[int] = field(default_factory=list)
    window_days: Optional[int] = None
    window_label: Optional[str] = None
    verification_status: Optional[str] = None

    def _counts(self) -> Dict[str, int]:
        c, n = self.cumulative, self.total_patients
        return {
            "attack-free (100%)": c[4],
            "90-99% reduction": c[3] - c[4],
            "70-89% reduction": c[2] - c[3],
            "50-69% reduction": c[1] - c[2],
            "<50% reduction (non-responder)": n - c[1],
        }

    @property
    def is_point_identified(self) -> bool:
        return sorted(self.cumulative) == [1, 2, 3, 4]

    @property
    def counts(self) -> Dict[str, int]:
        if not self.is_point_identified:
            raise ValueError(
                f"{self.trial}/{self.arm}: thresholds {sorted(self.cumulative)} do not "
                f"identify the exclusive categories; use `identified_set()` instead."
            )
        return self._counts()

    @property
    def proportions(self) -> Dict[str, float]:
        return {k: v / self.total_patients for k, v in self.counts.items()}

    def validate(self) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for lvl, val in self.cumulative.items():
            if val > self.total_patients:
                errors.append(f"level {lvl} cumulative {val} exceeds n={self.total_patients}")
            if val < 0:
                errors.append(f"level {lvl} has a negative count {val}")
        levels = sorted(self.cumulative)
        for a, b in zip(levels, levels[1:]):
            if self.cumulative[a] < self.cumulative[b]:
                errors.append(
                    f"nested thresholds are not monotone: {self.cumulative} "
                    f"(level {a} < level {b})"
                )
        if self.is_point_identified:
            total = sum(self.counts.values())
            if total != self.total_patients:
                errors.append(f"categories sum to {total}, expected {self.total_patients}")
        return len(errors) == 0, errors

    # -- partial identification -------------------------------------------
    def identified_set(self) -> pd.DataFrame:
        """Min / max of each exclusive category consistent with the reported counts.

        Solved with linear programming over the constraints
        ``sum(x_L..x_4) = c_L`` for every disclosed level ``L`` and ``sum(x) = n``.
        """
        return category_bounds(self.total_patients, self.cumulative)

    # -- sampling uncertainty ---------------------------------------------
    def proportion_intervals(self, alpha: float = 0.05) -> pd.DataFrame:
        """Clopper-Pearson exact intervals for each *disclosed* cumulative proportion."""
        rows = []
        for lvl in sorted(self.cumulative):
            k, n = self.cumulative[lvl], self.total_patients
            lo, hi = clopper_pearson(k, n, alpha)
            rows.append({"threshold": LEVELS[lvl], "level": lvl, "numerator": k,
                         "denominator": n, "proportion": k / n,
                         "ci_low": lo, "ci_high": hi,
                         "ci_coverage": 1 - alpha,
                         "interval_type": "exact binomial (sampling only)"})
        return pd.DataFrame(rows)

    def window_note(self) -> str:
        days = self.window_days
        label = self.window_label or "unspecified window"
        if not days:
            return f"Observation window: {label}. Window length is a design choice, not a " \
                   f"sampling uncertainty, and is reported separately from the confidence " \
                   f"interval."
        return (f"Observation window: {label} ({days:.0f} days). An attack-free count over "
                f"{days/30.44:.1f} months cannot be read as an annual or lifetime "
                f"probability; the window is not part of the confidence interval.")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Two-sided exact binomial (Clopper-Pearson) confidence interval."""
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def reconstruct_distribution(total_patients: int, cumulative: Dict[int, int], **meta
                             ) -> ResponseDistribution:
    """Validate nested thresholds and build a distribution object."""
    if not cumulative:
        raise ValueError("At least one cumulative threshold count is required.")
    for lvl, val in cumulative.items():
        if lvl not in LEVELS:
            raise ValueError(f"Unknown threshold level {lvl}; expected one of {sorted(LEVELS)}.")
        if val < 0 or val > total_patients:
            raise ValueError(
                f"Cumulative count for {LEVELS[lvl]} is {val} with n={total_patients}; "
                "counts must lie in [0, n]."
            )
    levels = sorted(cumulative)
    for a, b in zip(levels, levels[1:]):
        if cumulative[a] < cumulative[b]:
            raise ValueError(
                f"Nested thresholds violated: {LEVELS[a]}={cumulative[a]} < "
                f"{LEVELS[b]}={cumulative[b]}. Cumulative counts must be non-increasing "
                "in the threshold."
            )
    return ResponseDistribution(trial=meta.get("trial", ""), arm=meta.get("arm", ""),
                                total_patients=total_patients, cumulative=dict(cumulative),
                                observed_levels=levels,
                                window_days=meta.get("window_days"),
                                window_label=meta.get("window_label"),
                                verification_status=meta.get("verification_status"))


def category_bounds(n: int, cumulative: Dict[int, int]) -> pd.DataFrame:
    """Identified set (min/max) of each exclusive category given partial disclosure."""
    levels = sorted(cumulative)
    categories = list(reversed(CATEGORY_ORDER))  # x_0 (non-responder) first
    n_cat = len(categories)

    # objective for each category
    rows = []
    for idx, name in enumerate(categories):
        c = np.zeros(n_cat)
        c[idx] = 1.0

        # constraints: A_eq x = b_eq
        A, b = [], []
        # sum over all categories = n
        A.append(np.ones(n_cat)); b.append(float(n))
        # for each disclosed level L: sum of categories L..4 (i.e. the L-1 strictest... )
        # categories index 0 = "<50%" (level 0), 1 = 50-69, 2 = 70-89, 3 = 90-99, 4 = attack-free
        for lvl in levels:
            row = np.zeros(n_cat)
            for j in range(lvl, n_cat):     # category j >= lvl contributes
                row[j] = 1.0
            A.append(row); b.append(float(cumulative[lvl]))

        bounds = [(0, float(n)) for _ in range(n_cat)]
        res_min = optimize.linprog(c, A_eq=np.array(A), b_eq=np.array(b),
                                   bounds=bounds, method="highs")
        res_max = optimize.linprog(-c, A_eq=np.array(A), b_eq=np.array(b),
                                   bounds=bounds, method="highs")
        if not (res_min.success and res_max.success):
            lo = hi = float("nan")
        else:
            lo, hi = float(res_min.fun), float(-res_max.fun)
        rows.append({"category": name, "min": lo, "max": hi,
                     "point_identified": bool(np.isclose(lo, hi)),
                     "point_value": lo if np.isclose(lo, hi) else np.nan})
    out = pd.DataFrame(rows).set_index("category").loc[CATEGORY_ORDER].reset_index()
    out["min_share"] = out["min"] / n
    out["max_share"] = out["max"] / n
    return out


# ---------------------------------------------------------------------------
# Trial-facing helpers
# ---------------------------------------------------------------------------

_LEVEL_KEYWORDS = [
    (4, "attack-free"),
    (3, "90"),
    (2, "70"),
    (1, "50"),
]


def _level_from_text(text: str) -> Optional[int]:
    t = str(text).lower().replace(">=", "").replace(" ", "")
    if "attack-free" in t or "attackfree" in t:
        return 4
    for lvl, key in _LEVEL_KEYWORDS[1:]:
        if f"{key}%" in t:
            return lvl
    return None


def distributions_for_trial(trial: str, df: Optional[pd.DataFrame] = None
                            ) -> Dict[str, ResponseDistribution]:
    """Build one ResponseDistribution per treatment arm of a trial.

    Only participant counts are eligible. Trials that published percentages or
    mean rates without responder counts raise ``ValueError`` so the caller can
    say so rather than silently reconstructing something.
    """
    df = df if df is not None else get_clinical_trials()
    sub = df[(df["trial"] == trial) & df["is_count"]].copy()
    if sub.empty:
        raise ValueError(
            f"No participant-level responder counts are published for '{trial}'. "
            f"Only mean rates or percentages are available, so no exclusive response "
            f"distribution can be reconstructed."
        )

    out: Dict[str, ResponseDistribution] = {}
    for arm, grp in sub.groupby("treatment_arm"):
        cumulative: Dict[int, int] = {}
        for _, r in grp.iterrows():
            lvl = _level_from_text(r["endpoint_definition"])
            if lvl is None:
                continue
            cumulative[lvl] = int(r["numerator"])
        if not cumulative:
            continue
        n = int(grp["sample_size"].max())
        window_label = "; ".join(sorted(set(grp["measurement_window"].dropna())))
        window_days = float(grp["measurement_window_days"].max())
        status = "; ".join(sorted(set(grp["verification_status"])))
        out[arm] = reconstruct_distribution(
            n, cumulative, trial=trial, arm=arm,
            window_days=window_days if window_days else None,
            window_label=window_label or None, verification_status=status)
    if not out:
        raise ValueError(f"'{trial}' has counts but no recognised nested threshold endpoint.")
    return out


def available_trials(df: Optional[pd.DataFrame] = None) -> List[str]:
    """Trials with at least one participant-count responder endpoint."""
    df = df if df is not None else get_clinical_trials()
    trials = []
    for t in sorted(df.loc[df["is_count"], "trial"].unique()):
        try:
            distributions_for_trial(t, df)
        except ValueError:
            continue
        trials.append(t)
    return trials


# ---------------------------------------------------------------------------
# Partial identification of undisclosed subtypes
# ---------------------------------------------------------------------------

@dataclass
class SubtypeBounds:
    """Logical bounds on subtype-specific counts (not a confidence interval)."""

    n_type12: int
    n_other: int
    total_attack_free: int
    min_type12_attack_free: int
    max_type12_attack_free: int
    min_type12_proportion: float
    max_type12_proportion: float
    feasible_assignments: List[Tuple[int, int]]

    @property
    def width(self) -> float:
        return self.max_type12_proportion - self.min_type12_proportion


def enumerate_subtype_assignments(n_type12: int, n_other: int,
                                  total_attack_free: int) -> SubtypeBounds:
    """Enumerate every feasible split of an aggregate count across subtypes.

    Partial identification, not estimation: subtype-specific outcomes were not
    disclosed, so the data only constrain them to an interval.
    """
    if n_type12 < 0 or n_other < 0 or total_attack_free < 0:
        raise ValueError("Counts must be non-negative.")
    if total_attack_free > n_type12 + n_other:
        raise ValueError(
            f"total_attack_free={total_attack_free} exceeds the population "
            f"{n_type12 + n_other}; the assignment set is empty."
        )
    feasible = []
    lo = max(0, total_attack_free - n_other)
    hi = min(n_type12, total_attack_free)
    for t12 in range(lo, hi + 1):
        feasible.append((t12, total_attack_free - t12))
    return SubtypeBounds(
        n_type12=n_type12, n_other=n_other, total_attack_free=total_attack_free,
        min_type12_attack_free=lo, max_type12_attack_free=hi,
        min_type12_proportion=lo / n_type12 if n_type12 else float("nan"),
        max_type12_proportion=hi / n_type12 if n_type12 else float("nan"),
        feasible_assignments=feasible)


# ---------------------------------------------------------------------------
# Mean vs attack-free, window mechanics
# ---------------------------------------------------------------------------

def explain_average_vs_attackfree() -> str:
    return (
        "**Average attack-rate reduction** and **attack-free probability** answer different "
        "questions and are not interchangeable.\n\n"
        "- *Average reduction* is a ratio of mean rates. It can be dominated by patients with "
        "high baseline attack rates and says nothing about how many patients achieve complete "
        "control.\n"
        "- *Attack-free* is a binary count over a stated window. It is clinically legible but "
        "mechanically declines as the window lengthens, even when the underlying rate is "
        "unchanged.\n\n"
        "A therapy can post a large average reduction while few patients are attack-free, and "
        "the reverse cannot happen. Reporting both, with their windows, is what lets the two "
        "be compared across trials."
    )


def attack_free_probability(mean_per_month: float, dispersion: float = np.inf,
                            window_days: float = 168.0) -> float:
    """Probability of zero attacks over a window (analytic, deterministic).

    ``dispersion = inf`` gives a Poisson process; finite values give a
    gamma-mixed (negative-binomial) rate that represents patient-to-patient
    heterogeneity in the same underlying mean.
    """
    if mean_per_month <= 0:
        return 1.0
    lam = mean_per_month * (window_days / 30.4375)
    if np.isinf(dispersion):
        return float(np.exp(-lam))
    if dispersion <= 0:
        raise ValueError("dispersion must be positive (use np.inf for Poisson).")
    # Gamma-Poisson mixture: P(0) = (k / (k + lam))^k
    return float((dispersion / (dispersion + lam)) ** dispersion)


def window_sensitivity(mean_per_month: float = 1.93,
                       dispersions: Sequence[float] = (np.inf, 10.0, 3.0, 1.0),
                       windows_days: Sequence[float] = (28, 84, 168, 182, 365)
                       ) -> pd.DataFrame:
    """Analytic attack-free probability by window length and heterogeneity.

    ILLUSTRATIVE MODEL OUTPUT - not adjusted clinical evidence.
    """
    rows = []
    for disp in dispersions:
        label = "Poisson (no heterogeneity)" if np.isinf(disp) else f"gamma-mixed (k={disp:g})"
        for w in windows_days:
            rows.append({"rate_model": label, "dispersion": disp, "window_days": w,
                         "window_weeks": w / 7.0,
                         "attack_free_probability": attack_free_probability(
                             mean_per_month, disp, w),
                         "label": "ILLUSTRATIVE ASSUMPTION - NOT CLINICAL EVIDENCE"})
    return pd.DataFrame(rows)


def sample_size_resolution(n: int) -> Dict[str, float]:
    """The coarsest possible reading of an event rate: one patient = one step."""
    return {
        "n": n,
        "one_patient_share": 1.0 / n,
        "wilson_half_width_at_half": 1.96 * np.sqrt(0.25 / n),
        "note": (f"With n={n}, the reported rate moves in steps of {100/n:.1f} percentage "
                 f"points per patient. This is sampling resolution, not the observation "
                 f"window, and not a source of bias."),
    }
