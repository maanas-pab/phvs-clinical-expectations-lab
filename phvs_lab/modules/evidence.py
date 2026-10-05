"""Evidence layer: provenance reporting and independent re-verification.

Values are only marked *verified* after they have been matched against a
primary source that is re-downloaded at run time (ClinicalTrials.gov
structured results).  Nothing here fabricates patient-level data: only the
published aggregate counts are compared.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from phvs_lab.modules.data_loader import (
    get_clinical_trials,
    get_pitch_claim_audit,
    get_sources,
    validate_clinical_inputs,
)

CTG_API = "https://clinicaltrials.gov/api/v2/studies/{nct}"

# Endpoint keywords used to locate the matching registry outcome measure.
_ENDPOINT_PATTERNS = [
    (4, re.compile(r"attack[- ]free", re.I)),
    (3, re.compile(r"90\s*%", re.I)),
    (2, re.compile(r"70\s*%", re.I)),
    (1, re.compile(r"50\s*%", re.I)),
]


def endpoint_level(text: str) -> Optional[int]:
    """Map an endpoint definition onto its nested threshold level (1-4)."""
    for level, pattern in _ENDPOINT_PATTERNS:
        if pattern.search(str(text)):
            return level
    return None


LEVEL_LABELS = {
    1: ">=50% reduction",
    2: ">=70% reduction",
    3: ">=90% reduction",
    4: "attack-free",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_DOSE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|mcg|ug|µg|g|ml|iu|units?)\b", re.I)


def _doses(s: str) -> Tuple[str, ...]:
    return tuple(sorted(m.group(0).lower().replace(" ", "") for m in _DOSE_RE.finditer(str(s))))


def _arm_matches(candidate: str, target: str) -> float:
    """0-1 similarity between a registry group label and our arm label.

    A different strength (10 mg vs 20 mg) is a different arm and never matches.
    """
    a, b = _norm(candidate), _norm(target)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ca, cb = _doses(candidate), _doses(target)
    if ca and cb and ca != cb:
        return 0.0
    if a in b or b in a:
        return 0.9
    ta = set(re.findall(r"[a-z0-9]+", str(candidate).lower()))
    tb = set(re.findall(r"[a-z0-9]+", str(target).lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def extract_registry_counts(study: Dict) -> List[Dict]:
    """Flatten a ClinicalTrials.gov results section into comparable counts."""
    out: List[Dict] = []
    results = study.get("resultsSection") or {}
    om = results.get("outcomeMeasuresModule", {}).get("outcomeMeasures", [])
    for measure in om:
        groups = {g["id"]: g.get("title", "") for g in measure.get("groups", [])}
        base = {"measure_title": measure.get("title", ""),
                "units": measure.get("unitOfMeasure", ""),
                "param": measure.get("paramType", "")}

        def _emit(categories: List[Dict], cls_title: str) -> None:
            for cat in categories or []:
                cat_title = cat.get("title") or ""
                for m in cat.get("measurements", []) or []:
                    out.append({**base,
                                "class_title": cls_title,
                                "category_title": cat_title,
                                "group_title": groups.get(m.get("groupId"), ""),
                                "value": m.get("value")})

        classes = measure.get("classes") or []
        for cls in classes:
            _emit(cls.get("categories") or [], cls.get("title") or "")
        if not classes:
            _emit(measure.get("categories") or [], "")
    return out


_THRESHOLD_PATTERNS = {
    1: re.compile(r"(50)\s*%", re.I),
    2: re.compile(r"(70)\s*%", re.I),
    3: re.compile(r"(90)\s*%", re.I),
    4: re.compile(r"attack[- ]free", re.I),
}


def _find_registry_value(rows: List[Dict], arm: str, level: int) -> List[Dict]:
    """Return registry measurements that match an arm and a threshold level.

    Threshold text is matched against the *class* title when the registry
    provides one (so a measure that lists ">=50%, >=70% and >=90%" in its title
    is not mistaken for all three levels at once), otherwise against the measure
    title.  Only participant-count parameters are eligible.
    """
    pattern = _THRESHOLD_PATTERNS[level]
    hits = []
    for r in rows:
        if r["value"] is None or str(r["value"]).strip().upper() in {"", "NA", "N/A"}:
            continue
        param = str(r.get("param") or "").upper()
        if param and not any(k in param for k in ("COUNT", "NUMBER_OF", "PROPORTION", "PERCENT")):
            continue
        scope = r["class_title"] if r["class_title"] else f"{r['measure_title']} {r['category_title']}"
        if not pattern.search(scope):
            continue
        if not r["group_title"] or _arm_matches(r["group_title"], arm) < 0.6:
            continue
        hits.append(r)
    return hits


def _to_float(value) -> Optional[float]:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def verify_against_registry(df: Optional[pd.DataFrame] = None,
                            timeout: float = 15.0,
                            allow_network: bool = True) -> pd.DataFrame:
    """Re-download registry results and compare them with the stored counts.

    Returns a DataFrame with one row per clinical observation that carries a
    numerator, plus the columns ``registry_value``, ``match`` and ``detail``.
    Observations that are not registry-backed are reported as ``not_applicable``
    so that provisional pitch values stay visibly unverified.
    """
    df = df if df is not None else get_clinical_trials()
    target = df[df["is_count"] & df["nct_id"].notna()].copy()

    records: List[Dict] = []
    cache: Dict[str, Optional[List[Dict]]] = {}

    for _, row in target.iterrows():
        nct = row["nct_id"]
        level = endpoint_level(row["endpoint_definition"])
        rec = {
            "observation_id": row["observation_id"],
            "trial": row["trial"],
            "nct_id": nct,
            "arm": row["treatment_arm"],
            "endpoint": row["endpoint_definition"],
            "stored_value": row["numerator"],
            "verification_status": row["verification_status"],
            "registry_value": pd.NA,
            "match": "not_checked",
            "detail": "",
        }

        if level is None:
            rec["match"] = "not_applicable"
            rec["detail"] = "Endpoint is not a nested threshold endpoint."
            records.append(rec)
            continue

        if nct not in cache:
            if not allow_network:
                cache[nct] = None
            else:
                try:
                    import requests
                    resp = requests.get(CTG_API.format(nct=nct), timeout=timeout)
                    if resp.status_code == 200:
                        cache[nct] = extract_registry_counts(resp.json())
                    else:
                        cache[nct] = None
                except Exception:
                    cache[nct] = None
            time.sleep(0.2)

        rows = cache[nct]
        if rows is None:
            rec["match"] = "network_unavailable"
            rec["detail"] = "ClinicalTrials.gov could not be reached; stored value left as-is."
            records.append(rec)
            continue

        hits = _find_registry_value(rows, row["treatment_arm"], level)
        if not hits:
            rec["match"] = "not_found_in_registry"
            rec["detail"] = ("No matching arm/threshold measurement in the registry results "
                             "record; value remains unverified.")
        else:
            stored = float(row["numerator"])
            n = float(row["denominator"])
            pct_equivalent = 100.0 * stored / n if n else float("nan")
            candidates = [(h, _to_float(h["value"])) for h in hits]
            candidates = [(h, v) for h, v in candidates if v is not None]
            rec["registry_value"] = candidates[0][1] if candidates else pd.NA
            matched = next((v for _, v in candidates if abs(v - stored) < 1e-9), None)
            form = "count"
            if matched is None and n:
                matched = next((v for _, v in candidates
                                if abs(v - pct_equivalent) <= 0.15), None)
                if matched is not None:
                    rec["registry_value"] = matched
                    form = "percent"
            if matched is not None:
                rec["match"] = "verified" if form == "count" else "verified_percent_form"
                rec["detail"] = (
                    f"Registry {'count' if form == 'count' else 'percentage'} {matched:g} "
                    f"matches the stored value {stored:g} for arm "
                    f"'{hits[0]['group_title']}'."
                )
            else:
                rec["match"] = "mismatch"
                rec["detail"] = (f"Registry reports "
                                 f"{[v for _, v in candidates]} but the file stores "
                                 f"{stored:g}.")
        records.append(rec)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def provenance_summary(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Counts of observations by value_type x verification_status."""
    df = df if df is not None else get_clinical_trials()
    counts = (df.groupby(["value_type", "verification_status"])
                .agg(observations=("observation_id", "count"),
                     trials=("trial", "nunique"))
                .reset_index())
    counts["share"] = counts["observations"] / counts["observations"].sum()
    return counts.sort_values(["value_type", "verification_status"]).reset_index(drop=True)


def audit_report(df: Optional[pd.DataFrame] = None) -> Dict:
    """Full structural + provenance audit of the evidence layer.

    Returns ``{"findings": [...], "provenance": DataFrame,
    "claims": DataFrame, "source_coverage": DataFrame, "ok": bool}``.
    """
    df = df if df is not None else get_clinical_trials()
    findings = validate_clinical_inputs(df)

    sources = get_sources()
    used_ids = set(df["source_id"].dropna())
    known_ids = set(sources["source_id"].dropna())
    for sid in sorted(used_ids - known_ids):
        findings.append({"type": "unknown_source", "id": sid,
                         "message": f"Observations reference source_id '{sid}' which is not "
                                    f"registered in sources.csv."})

    missing_url = sources[sources["url"].isna() | (sources["url"].astype(str).str.strip() == "")]
    for _, r in missing_url.iterrows():
        findings.append({"type": "source_without_url", "id": r["source_id"],
                         "message": f"{r['source_id']} has no URL."})

    coverage = (sources.assign(used=sources["source_id"].isin(used_ids))
                       [["source_id", "title", "publisher", "source_type", "url",
                         "publication_date", "used"]])
    return {
        "findings": findings,
        "provenance": provenance_summary(df),
        "claims": get_pitch_claim_audit(),
        "source_coverage": coverage,
        "ok": len(findings) == 0,
    }
