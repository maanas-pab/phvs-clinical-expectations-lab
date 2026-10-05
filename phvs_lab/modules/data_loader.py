"""Auditable evidence layer: CSV loading, schema validation and structural checks.

Source facts and analyst assumptions are kept in separate files and are tagged
row-by-row with ``value_type`` (reported / calculated / assumed) and
``verification_status``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phvs_lab.config import DATA_DIR

CLINICAL_COLUMNS = [
    "observation_id", "drug", "trial", "nct_id", "treatment_arm", "population",
    "sample_size", "endpoint_definition", "endpoint_type", "measurement_window",
    "measurement_window_days", "numerator", "denominator", "unit",
    "background_treatment", "exclusions", "source_id", "source_url",
    "document_page", "publication_date", "value_type", "verification_status",
    "verified_by", "verified_date", "notes",
]

ASSUMPTION_COLUMNS = [
    "parameter", "value", "unit", "category", "source_id", "source_url",
    "assumption_type", "verification_status", "notes",
]

SOURCE_COLUMNS = [
    "source_id", "title", "publisher", "source_type", "url",
    "publication_date", "retrieval_date", "used_for", "notes",
]

CLAIM_COLUMNS = [
    "claim_id", "claim_source", "claim_text", "claim_value", "verified_value",
    "verification_status", "source_id", "source_url", "impact_on_thesis",
]

VALID_VALUE_TYPES = {"reported", "calculated", "assumed"}
VALID_VERIFICATION = {
    "verified_primary", "verified_secondary", "corroborated_secondary",
    "provisional_unverified", "unverified",
}

_lock = threading.Lock()
_cache: Dict[str, pd.DataFrame] = {}


class EvidenceSchemaError(ValueError):
    """Raised when an input file does not satisfy the documented schema."""


def _require(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise EvidenceSchemaError(f"Missing required columns in {name}: {missing}")


def _read(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise EvidenceSchemaError(f"Input file not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_clinical_trials() -> pd.DataFrame:
    """Load and type the clinical observation table."""
    df = _read("clinical_trials.csv")
    _require(df, CLINICAL_COLUMNS, "clinical_trials.csv")

    for col in ["sample_size", "numerator", "denominator", "measurement_window_days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_count"] = df["numerator"].notna() & df["denominator"].notna()
    # Counts become proportions directly; reported percentages are on a 0-100 scale.
    df["proportion"] = np.where(
        df["is_count"], df["numerator"] / df["denominator"],
        np.where(df["unit"].eq("percentage"), df["numerator"] / 100.0, np.nan),
    )
    df["value_type"] = df["value_type"].str.strip().str.lower()
    df["verification_status"] = df["verification_status"].str.strip()
    bad_type = set(df["value_type"].dropna()) - VALID_VALUE_TYPES
    if bad_type:
        raise EvidenceSchemaError(f"Unknown value_type in clinical_trials.csv: {sorted(bad_type)}")

    df["is_verified"] = df["verification_status"].isin(
        ["verified_primary", "verified_secondary", "corroborated_secondary"]
    )
    return df


def count_observations(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Observations recorded as participant counts (the only ones usable as counts)."""
    df = df if df is not None else get_clinical_trials()
    return df[df["is_count"]].copy()


def load_commercial_assumptions() -> pd.DataFrame:
    """Load commercial / valuation assumptions (analyst-editable inputs)."""
    df = _read("commercial_assumptions.csv")
    _require(df, ASSUMPTION_COLUMNS, "commercial_assumptions.csv")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["value"].isna().any():
        bad = df.loc[df["value"].isna(), "parameter"].tolist()
        raise EvidenceSchemaError(f"Non-numeric assumption values: {bad}")
    return df


def load_valuation_model() -> pd.DataFrame:
    """Load the pitch's annual cash-flow table (imported, unmodified)."""
    df = _read("valuation_model.csv")
    _require(df, ["year", "fcf_usd", "pv_fcf_usd", "cumulative_pv_usd"], "valuation_model.csv")
    for col in df.columns:
        if col in ("year", "notes"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_sources() -> pd.DataFrame:
    df = _read("sources.csv")
    _require(df, SOURCE_COLUMNS, "sources.csv")
    return df


def load_pitch_claim_audit() -> pd.DataFrame:
    df = _read("pitch_claim_audit.csv")
    _require(df, CLAIM_COLUMNS, "pitch_claim_audit.csv")
    return df


# ---------------------------------------------------------------------------
# Cached accessors
# ---------------------------------------------------------------------------

def _cached(key: str, loader) -> pd.DataFrame:
    with _lock:
        if key not in _cache:
            _cache[key] = loader()
        return _cache[key]


def get_clinical_trials() -> pd.DataFrame:
    return _cached("clinical", load_clinical_trials)


def get_commercial_assumptions() -> pd.DataFrame:
    return _cached("commercial", load_commercial_assumptions)


def get_valuation_model() -> pd.DataFrame:
    return _cached("valuation", load_valuation_model)


def get_sources() -> pd.DataFrame:
    return _cached("sources", load_sources)


def get_pitch_claim_audit() -> pd.DataFrame:
    return _cached("claims", load_pitch_claim_audit)


def reload_all() -> None:
    """Force a reload of every input file (used after an in-app edit)."""
    with _lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Assumption access
# ---------------------------------------------------------------------------

def get_assumption(name: str, df: Optional[pd.DataFrame] = None) -> float:
    """Return a single assumption value by parameter name."""
    df = df if df is not None else get_commercial_assumptions()
    row = df.loc[df["parameter"] == name]
    if row.empty:
        raise KeyError(f"Assumption '{name}' not found in commercial_assumptions.csv")
    val = row["value"].iloc[0]
    if pd.isna(val):
        raise ValueError(f"Assumption '{name}' has a non-numeric value")
    return float(val)


def assumption_meta(name: str, df: Optional[pd.DataFrame] = None) -> Dict:
    """Return the provenance metadata for an assumption."""
    df = df if df is not None else get_commercial_assumptions()
    row = df.loc[df["parameter"] == name]
    if row.empty:
        raise KeyError(f"Assumption '{name}' not found in commercial_assumptions.csv")
    return row.iloc[0].to_dict()


def assumptions_by_category(df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    df = df if df is not None else get_commercial_assumptions()
    return {k: g.copy() for k, g in df.groupby("category", sort=True)}


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def endpoint_level(text: str) -> Optional[int]:
    """Nested-threshold level of an endpoint definition (1 => >=50%, 4 => attack-free)."""
    t = str(text).lower()
    if "attack-free" in t or "attack free" in t:
        return 4
    for key, lvl in (("90%", 3), ("70%", 2), ("50%", 1)):
        if key in t:
            return lvl
    return None


def endpoint_family(text: str) -> str:
    """Coarse endpoint family, so differently worded definitions can still be linked."""
    lvl = endpoint_level(text)
    if lvl is not None:
        return {1: ">=50% responder", 2: ">=70% responder", 3: ">=90% responder",
                4: "attack-free"}[lvl]
    t = str(text).lower()
    if "time to first" in t:
        return "time to first attack"
    if "acute" in t or "requiring" in t:
        return "acute-treated attacks"
    if "moderate" in t or "severe" in t:
        return "moderate/severe attacks"
    if "mean" in t or "least-squares" in t or "normalized" in t or "rate" in t:
        return "mean attack rate"
    if "percentage" in t or "proportion" in t:
        return "other responder proportion"
    return "other"


def validate_threshold_nesting(df: Optional[pd.DataFrame] = None) -> List[Dict]:
    """Check that threshold counts nest correctly within each (trial, arm).

    For a set of responder endpoints measured on the same arm and window the
    counts must satisfy  n(>=50) >= n(>=70) >= n(>=90) >= n(attack-free).
    """
    df = df if df is not None else get_clinical_trials()
    findings: List[Dict] = []
    resp = df[df["is_count"]].copy()
    resp["level"] = resp["endpoint_definition"].map(endpoint_level)
    resp = resp[resp["level"].notna()]

    for (trial, arm, window), grp in resp.groupby(["trial", "treatment_arm", "measurement_window"]):
        counts = dict(zip(grp["level"], grp["numerator"]))
        ordered = [counts[k] for k in sorted(counts) if k in counts]
        if len(ordered) >= 2 and any(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1)):
            findings.append({
                "type": "threshold_nesting_violation",
                "trial": trial, "arm": arm, "window": window,
                "counts": counts,
                "message": f"{trial} / {arm}: nested threshold counts are not monotone "
                           f"({counts}).",
            })
        denom_unique = grp["denominator"].unique()
        if len(denom_unique) > 1:
            findings.append({
                "type": "inconsistent_denominator",
                "trial": trial, "arm": arm, "window": window,
                "counts": dict(zip(grp["level"], grp["denominator"])),
                "message": f"{trial} / {arm}: responder endpoints use different denominators "
                           f"{sorted(denom_unique)}.",
            })
    return findings


def validate_clinical_inputs(df: Optional[pd.DataFrame] = None) -> List[Dict]:
    """Run every structural check on the clinical observation table."""
    df = df if df is not None else get_clinical_trials()
    findings: List[Dict] = []

    if df["observation_id"].duplicated().any():
        dupes = df.loc[df["observation_id"].duplicated(), "observation_id"].tolist()
        findings.append({"type": "duplicate_id", "ids": dupes,
                         "message": f"Duplicate observation_id values: {dupes}"})

    num, den, n = df["numerator"], df["denominator"], df["sample_size"]
    bad = df[num.notna() & den.notna() & (num > den)]
    for _, r in bad.iterrows():
        findings.append({"type": "numerator_exceeds_denominator",
                         "id": r["observation_id"],
                         "message": f"{r['observation_id']}: numerator {r['numerator']} > "
                                    f"denominator {r['denominator']}."})

    bad = df[den.notna() & n.notna() & (den > n)]
    for _, r in bad.iterrows():
        findings.append({"type": "denominator_exceeds_sample_size",
                         "id": r["observation_id"],
                         "message": f"{r['observation_id']}: denominator {r['denominator']} > "
                                    f"sample size {r['sample_size']}."})

    bad = df[num.notna() & num.lt(0)]
    for _, r in bad.iterrows():
        findings.append({"type": "negative_count", "id": r["observation_id"],
                         "message": f"{r['observation_id']}: negative numerator."})

    # Unit / value consistency: participant counts need a denominator; percentages
    # are recorded on a 0-100 scale and must not be read as counts.
    for _, r in df[num.notna() & df["unit"].eq("participants") & den.isna()].iterrows():
        findings.append({"type": "count_without_denominator", "id": r["observation_id"],
                         "message": f"{r['observation_id']}: participant count "
                                    f"{r['numerator']} has no denominator."})
    for _, r in df[den.notna() & num.isna()].iterrows():
        findings.append({"type": "denominator_without_numerator", "id": r["observation_id"],
                         "message": f"{r['observation_id']}: denominator recorded without a "
                                    f"numerator."})
    pct_rows = df[df["unit"].eq("percentage") & num.notna()]
    bad = pct_rows[(pct_rows["numerator"] < 0) | (pct_rows["numerator"] > 100)]
    for _, r in bad.iterrows():
        findings.append({"type": "percentage_out_of_range", "id": r["observation_id"],
                         "message": f"{r['observation_id']}: percentage "
                                    f"{r['numerator']} outside 0-100."})
    bad = df[df["unit"].eq("percentage") & df["is_count"]]
    for _, r in bad.iterrows():
        findings.append({"type": "percentage_recorded_as_count", "id": r["observation_id"],
                         "message": f"{r['observation_id']}: percentage is paired with a "
                                    f"denominator, so it could be misread as a count."})

    for _, r in df[df["source_url"].isna() | (df["source_url"].astype(str).str.strip() == "")].iterrows():
        findings.append({"type": "missing_source", "id": r["observation_id"],
                         "message": f"{r['observation_id']}: no source URL recorded."})

    for _, r in df[(df["value_type"] == "reported") & (~df["is_verified"])].iterrows():
        findings.append({"type": "unverified_reported_value", "id": r["observation_id"],
                         "message": f"{r['observation_id']}: marked 'reported' but not verified "
                                    f"against a primary source."})

    # Cross-trial comparability (different windows / populations for the same
    # endpoint family) is reported by trial_audit.comparability_warnings, not as
    # a data error: it is a reading hazard rather than a schema defect.

    findings.extend(validate_threshold_nesting(df))
    return findings


def separate_facts_from_assumptions(df: Optional[pd.DataFrame] = None
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations into source facts and analyst assumptions."""
    df = df if df is not None else get_clinical_trials()
    facts = df[df["value_type"].isin(["reported", "calculated"])].copy()
    assumptions = df[df["value_type"] == "assumed"].copy()
    return facts, assumptions


def describe_trial_set(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per (trial, arm) with the design facts needed for comparison."""
    df = df if df is not None else get_clinical_trials()
    g = (df.groupby(["drug", "trial", "nct_id", "treatment_arm", "population"], dropna=False)
           .agg(sample_size=("sample_size", "max"),
                windows=("measurement_window", lambda s: "; ".join(sorted(set(s.dropna())))),
                endpoints=("endpoint_definition", "nunique"),
                first_source=("source_id", "first"),
                statuses=("verification_status", lambda s: "; ".join(sorted(set(s)))))
           .reset_index())
    return g
