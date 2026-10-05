"""Physician discrete-choice experiment (DCE) for PHVS Clinical Expectations Lab.

The module answers a narrow question: **which assumption in the lab is not yet
measured, and what instrument would measure it?**

Status: **survey not conducted.** Nothing in this module is a physician
response. It provides

* a reproducible vignette design (attributes, fractional factorial, choice tasks),
* a CSV contract for importing anonymised responses when they exist,
* a conditional-logit estimator that turns responses into part-worth utilities
  with standard errors and attribute importance,
* a synthetic demo mode that is labelled as such everywhere it appears.

The estimated utilities are the empirical counterpart of the ``mnl_*`` and
``oral_preference_*`` coefficients in the assumption ledger: those are analyst
assumptions today, and this instrument is how they would stop being assumptions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from phvs_lab.modules.data_loader import get_commercial_assumptions, get_assumption

__all__ = [
    "VignetteAttribute", "ClinicalVignette", "ChoiceTask", "DCE_ATTRIBUTES",
    "generate_full_factorial", "generate_fractional_factorial", "create_choice_tasks",
    "assign_tasks_to_respondents", "create_stage2_reveal", "export_survey_design",
    "import_survey_responses", "simulate_demo_responses", "fit_conditional_logit",
    "attribute_importance", "survey_status", "survey_maps_to_ledger",
    "describe_tasks", "demonstrate_survey_design", "explain_survey_limitations",
]

DEMO_NOTICE = "DEMO MODE - SYNTHETIC RESPONSES, NOT PHYSICIANS."
NOT_CONDUCTED_NOTICE = "SURVEY NOT CONDUCTED - no physician responses exist for PHVS."
RESPONSE_SCHEMA = ["respondent_id", "task_id", "choice", "stage",
                   "response_time_seconds", "physician_specialty",
                   "years_experience", "n_hae_patients_per_year"]


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------

@dataclass
class VignetteAttribute:
    """An attribute in the discrete-choice experiment."""
    name: str
    levels: List[str]
    description: str


@dataclass
class ClinicalVignette:
    """A single clinical vignette (treatment profile)."""
    profile_id: str
    attributes: Dict[str, str]
    route_disclosed: bool = False

    def to_dict(self) -> Dict:
        return {"profile_id": self.profile_id, "attributes": self.attributes,
                "route_disclosed": self.route_disclosed}


@dataclass
class ChoiceTask:
    """A single choice task (pair of vignettes)."""
    task_id: str
    vignette_a: ClinicalVignette
    vignette_b: ClinicalVignette
    forced_choice: bool = True
    opt_out_option: bool = False
    task_order: int = 0


DCE_ATTRIBUTES: List[VignetteAttribute] = [
    VignetteAttribute("attack_control",
                      ["50% reduction", "70% reduction", "90% reduction", "Attack-free"],
                      "Expected reduction in monthly HAE attacks"),
    VignetteAttribute("route",
                      ["Oral daily", "Subcutaneous every 2 weeks", "IV every 2 weeks"],
                      "Route of administration"),
    VignetteAttribute("dosing_frequency",
                      ["Daily", "Weekly", "Every 2 weeks", "Monthly"],
                      "Dosing frequency"),
    VignetteAttribute("safety_profile",
                      ["Well-established (10+ years)", "Moderate experience (3-5 years)",
                       "Limited experience (<2 years)"],
                      "Long-term safety data availability"),
    VignetteAttribute("out_of_pocket",
                      ["$0-50/month", "$50-200/month", "$200-500/month", ">$500/month"],
                      "Patient out-of-pocket cost per month"),
    VignetteAttribute("formulary_status",
                      ["Preferred tier", "Non-preferred tier", "Prior authorization required",
                       "Not on formulary"],
                      "Insurance formulary placement"),
]

ATTR_BY_NAME = {a.name: a for a in DCE_ATTRIBUTES}


def generate_full_factorial(attributes: Optional[List[VignetteAttribute]] = None
                            ) -> List[Dict]:
    """Generate all combinations of attribute levels."""
    attributes = attributes or DCE_ATTRIBUTES
    names = [a.name for a in attributes]
    return [dict(zip(names, combo))
            for combo in product(*[a.levels for a in attributes])]


def generate_fractional_factorial(
    attributes: Optional[List[VignetteAttribute]] = None,
    n_profiles: int = 32,
    seed: int = 42,
    n_candidates: int = 600,
) -> List[Dict]:
    """Balanced fractional factorial chosen for orthogonality.

    Random candidate subsets are drawn from the full factorial and scored on the
    smallest eigenvalue of their dummy-coded correlation matrix (E-optimality):
    the design that maximises it wins. This is a small, reproducible stand-in for
    a formal D-optimal search and prevents the rank deficiency and accidental
    collinearity that a naive "most balanced" selection produces.
    """
    attributes = attributes or DCE_ATTRIBUTES
    full = generate_full_factorial(attributes)
    n_profiles = int(min(n_profiles, len(full)))
    rng = np.random.default_rng(seed)

    X_full = _dummy_matrix(full, attributes)
    best_idx, best_score = None, np.inf
    draws = max(n_candidates, 1)
    for _ in range(draws):
        if len(full) == n_profiles:
            idx = np.arange(n_profiles)
        else:
            idx = rng.choice(len(full), n_profiles, replace=False)
        X = X_full[idx]
        Xz = X - X.mean(axis=0)
        norm = np.sqrt((Xz ** 2).sum(axis=0))
        constant = norm < 1e-12
        n_constant = int(constant.sum())
        safe = np.where(constant, 1.0, norm)
        Xs = Xz / safe
        G = Xs.T @ Xs
        G[constant, :] = 0.0
        G[:, constant] = 0.0
        np.fill_diagonal(G, 1.0)
        lam_min = float(np.linalg.eigvalsh(G).min())
        score = -lam_min + float(n_constant)
        if score < best_score - 1e-12:
            best_score, best_idx = score, np.asarray(idx)
        if score < best_score - 1e-12:
            best_score, best_idx = score, np.asarray(idx)

    if best_idx is None:                                    # pragma: no cover
        best_idx = rng.choice(len(full), n_profiles, replace=False)
    chosen = [{k: full[i][k] for k in full[best_idx[0]]} for i in best_idx]
    return chosen


def _dummy_matrix(profiles: List[Dict],
                  attributes: Optional[List[VignetteAttribute]] = None) -> np.ndarray:
    """Dummy coding with the first level of each attribute as reference."""
    attributes = attributes or DCE_ATTRIBUTES
    cols = []
    for a in attributes:
        for lv in a.levels[1:]:
            cols.append([1.0 if p[a.name] == lv else 0.0 for p in profiles])
    return np.asarray(cols, dtype=float).T


def create_choice_tasks(
    profiles: List[Dict],
    n_tasks_per_respondent: int = 12,
    n_respondents: int = 100,
    seed: int = 42,
) -> List[ChoiceTask]:
    """Create pairwise choice tasks from a set of profiles."""
    rng = np.random.default_rng(seed)
    for i, p in enumerate(profiles):
        p.setdefault("profile_id", f"P{i:02d}")

    pairs = [(profiles[i], profiles[j])
             for i in range(len(profiles)) for j in range(i + 1, len(profiles))]
    n_tasks = min(n_tasks_per_respondent * n_respondents, len(pairs))
    if n_tasks == 0:
        raise ValueError("Need at least two profiles to form a choice task.")
    selected = rng.choice(len(pairs), n_tasks, replace=False)

    tasks = []
    for idx, pair_idx in enumerate(selected):
        p1, p2 = pairs[pair_idx]
        tasks.append(ChoiceTask(
            task_id=f"T{idx:04d}",
            vignette_a=ClinicalVignette(p1["profile_id"], dict(p1), route_disclosed=False),
            vignette_b=ClinicalVignette(p2["profile_id"], dict(p2), route_disclosed=False),
        ))
    return tasks


def assign_tasks_to_respondents(
    tasks: List[ChoiceTask],
    n_respondents: int,
    tasks_per_respondent: int = 12,
    seed: int = 42,
) -> Dict[str, List[ChoiceTask]]:
    """Assign a balanced, order-randomised subset of tasks to each respondent."""
    rng = np.random.default_rng(seed)
    if tasks_per_respondent > len(tasks):
        raise ValueError(f"tasks_per_respondent={tasks_per_respondent} exceeds the "
                         f"{len(tasks)} tasks available.")
    assignments: Dict[str, List[ChoiceTask]] = {}
    for r in range(n_respondents):
        idx = rng.choice(len(tasks), tasks_per_respondent, replace=False)
        chosen = [tasks[i] for i in idx]
        rng.shuffle(chosen)
        for order, task in enumerate(chosen):
            task.task_order = order
        assignments[f"R{r:04d}"] = chosen
    return assignments


def create_stage2_reveal(tasks: List[ChoiceTask]) -> List[ChoiceTask]:
    """Same tasks with the route attribute disclosed (stage 2)."""
    return [
        ChoiceTask(
            task_id=t.task_id + "_S2",
            vignette_a=ClinicalVignette(t.vignette_a.profile_id,
                                        dict(t.vignette_a.attributes), route_disclosed=True),
            vignette_b=ClinicalVignette(t.vignette_b.profile_id,
                                        dict(t.vignette_b.attributes), route_disclosed=True),
            forced_choice=t.forced_choice, opt_out_option=t.opt_out_option,
        ) for t in tasks
    ]


def export_survey_design(tasks: List[ChoiceTask],
                         stage2_tasks: Optional[List[ChoiceTask]] = None,
                         assignments: Optional[Dict[str, List[ChoiceTask]]] = None,
                         output_path: Optional[str] = None) -> Dict:
    """Export the design as a JSON-serialisable dict (and optionally to disk)."""
    stage2_tasks = stage2_tasks or create_stage2_reveal(tasks)
    assignments = assignments or {}
    design = {
        "metadata": {
            "created": datetime.now(timezone.utc).isoformat(),
            "attributes": [{"name": a.name, "levels": a.levels,
                            "description": a.description} for a in DCE_ATTRIBUTES],
            "n_tasks": len(tasks),
            "n_respondents": len(assignments),
            "stage1_route_hidden": True,
            "stage2_route_revealed": True,
            "status": NOT_CONDUCTED_NOTICE,
        },
        "tasks": [t.vignette_a.to_dict() for t in tasks]
                 + [t.vignette_b.to_dict() for t in tasks],
        "stage2_tasks": [t.vignette_a.to_dict() for t in stage2_tasks]
                        + [t.vignette_b.to_dict() for t in stage2_tasks],
        "assignments": {rid: [t.task_id for t in tl] for rid, tl in assignments.items()},
    }
    if output_path:
        with open(output_path, "w") as fh:
            json.dump(design, fh, indent=2)
    return design


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

def survey_status(response_path: Optional[str] = None) -> Dict:
    """Status of the instrument: never presented as conducted data."""
    if response_path:
        try:
            df = import_survey_responses(response_path)
            return {"conducted": True, "mode": "fielded", "n_responses": len(df),
                    "n_respondents": int(df["respondent_id"].nunique()),
                    "notice": "FIELDED RESPONSES - imported from file, not generated here."}
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            return {"conducted": False, "mode": "error", "n_responses": 0,
                    "notice": f"Response file could not be read: {exc}"}
    return {"conducted": False, "mode": "design_only", "n_responses": 0,
            "n_respondents": 0, "notice": NOT_CONDUCTED_NOTICE}


def import_survey_responses(file_path: str) -> pd.DataFrame:
    """Import anonymised responses with validation; hash respondent IDs."""
    df = pd.read_csv(file_path)
    required = ["respondent_id", "task_id", "choice", "stage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["respondent_id"] = df["respondent_id"].astype(str)
    already_hashed = df["respondent_id"].str.match(r"^[a-f0-9]{12}$").all()
    if not already_hashed:
        df["respondent_id"] = df["respondent_id"].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12])

    df["choice"] = df["choice"].astype(str).str.strip().str.upper()
    bad = set(df["choice"]) - {"A", "B"}
    if bad:
        raise ValueError(f"choice must be A or B; found {sorted(bad)}")
    if not set(df["stage"].astype(int)) <= {1, 2}:
        raise ValueError("stage must be 1 (route hidden) or 2 (route disclosed)")
    return df


def simulate_demo_responses(tasks: List[ChoiceTask],
                            n_respondents: int = 100,
                            tasks_per_respondent: int = 12,
                            seed: int = 7) -> pd.DataFrame:
    """Synthetic responses from a stated parameter vector - DEMO ONLY.

    The generating coefficients are deliberately close to the analyst's ``mnl_*``
    assumptions so the demo shows what a fitted survey would look like, not what
    physicians said.
    """
    rng = np.random.default_rng(seed)
    design = task_design_matrix(tasks)
    beta_true = np.zeros(design.shape[1])

    def put(attr: str, level: str, value: float) -> None:
        col = f"{attr}__{level}"
        if col in design.columns:
            beta_true[design.columns.get_loc(col)] = value

    # reference levels are the first level of each attribute
    put("route", "Subcutaneous every 2 weeks", -0.9)   # oral is the reference
    put("route", "IV every 2 weeks", -1.2)
    put("attack_control", "70% reduction", 0.3)
    put("attack_control", "90% reduction", 0.6)
    put("attack_control", "Attack-free", 0.9)
    put("out_of_pocket", "$200-500/month", -0.4)
    put("out_of_pocket", ">$500/month", -0.9)
    put("safety_profile", "Limited experience (<2 years)", -0.5)
    put("formulary_status", "Not on formulary", -0.8)

    rows = []
    for r in range(n_respondents):
        idx = rng.choice(len(tasks), min(tasks_per_respondent, len(tasks)), replace=False)
        respondent_noise = rng.normal(0, 0.35, size=design.shape[1])
        for order, t_i in enumerate(idx):
            task = tasks[t_i]
            d = design.iloc[t_i].to_numpy(float) + respondent_noise
            p_a = 1.0 / (1.0 + np.exp(-d @ beta_true))
            choice = "A" if rng.random() < p_a else "B"
            rows.append({
                "respondent_id": f"R{r:04d}",
                "task_id": task.task_id,
                "choice": choice,
                "stage": 1,
                "response_time_seconds": float(np.round(rng.lognormal(3.4, 0.4), 1)),
                "physician_specialty": "allergist" if r % 3 else "immunologist",
                "years_experience": int(np.clip(round(rng.normal(14, 6)), 1, 40)),
                "n_hae_patients_per_year": int(np.clip(round(rng.lognormal(3.0, 0.6)), 1, 400)),
                "order": order,
            })
    out = pd.DataFrame(rows)
    out.attrs["label"] = DEMO_NOTICE
    return out


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def task_design_matrix(tasks: List[ChoiceTask]) -> pd.DataFrame:
    """Dummy-coded (A minus B) design matrix for a list of pairwise tasks."""
    cols: Dict[str, List[float]] = {}
    for a in DCE_ATTRIBUTES:
        for lv in a.levels[1:]:          # first level is the reference
            cols[f"{a.name}__{lv}"] = []
    for t in tasks:
        for name, levels in cols.items():
            attr, lv = name.split("__", 1)
            x_a = 1.0 if t.vignette_a.attributes.get(attr) == lv else 0.0
            x_b = 1.0 if t.vignette_b.attributes.get(attr) == lv else 0.0
            levels.append(x_a - x_b)
    return pd.DataFrame(cols)


def fit_conditional_logit(responses: pd.DataFrame,
                          tasks: List[ChoiceTask],
                          ridge: float = 0.05) -> Dict:
    """Fit a binary conditional logit (panel-free, main effects) to task choices.

    Returns part-worth utilities with standard errors, z-statistics and
    two-sided p-values. Respondent-level clustering is out of scope for the
    design stage; the standard errors are therefore optimistic and are labelled
    as such in the output.
    """
    design = task_design_matrix(tasks)
    id_to_idx = {t.task_id: i for i, t in enumerate(tasks)}
    usable = responses[responses["task_id"].isin(id_to_idx)].copy()
    if usable.empty:
        raise ValueError("No responses match the supplied task design.")
    usable["idx"] = usable["task_id"].map(id_to_idx)
    usable["y"] = (usable["choice"].str.upper() == "A").astype(float)

    X = design.iloc[usable["idx"].to_numpy()].to_numpy(float)
    y = usable["y"].to_numpy(float)

    def nll(beta: np.ndarray) -> float:
        eta = X @ beta
        return float(np.sum(np.logaddexp(0, eta) - y * eta)
                     + 0.5 * ridge * float(beta @ beta))

    def grad(beta: np.ndarray) -> np.ndarray:
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        return X.T @ (p - y) + ridge * beta

    beta0 = np.zeros(X.shape[1])
    res = optimize.minimize(nll, beta0, jac=grad, method="L-BFGS-B")
    beta = res.x
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1 - p)
    hessian = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
    cov = np.linalg.pinv(hessian)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))

    out = pd.DataFrame({
        "term": design.columns,
        "utility": beta, "std_error": se, "z": z, "p_value": pvals,
        "attribute": [c.split("__", 1)[0] for c in design.columns],
        "level": [c.split("__", 1)[1] for c in design.columns],
    })
    ll = -float(np.sum(np.logaddexp(0, eta) - y * eta))
    null_p = y.mean()
    ll0 = float(np.sum(y * np.log(max(null_p, 1e-12))
                       + (1 - y) * np.log(max(1 - null_p, 1e-12))))
    return {
        "coefficients": out,
        "n_responses": int(len(y)),
        "n_tasks_used": int(usable["task_id"].nunique()),
        "n_respondents": int(usable["respondent_id"].nunique()),
        "log_likelihood": ll,
        "null_log_likelihood": ll0,
        "mcfadden_r2": float(1 - ll / ll0) if ll0 else np.nan,
        "ridge": ridge,
        "se_note": "Main-effects conditional logit; standard errors are "
                   "respondent-cluster-naive and therefore optimistic.",
        "label": DEMO_NOTICE if str(responses.attrs.get("label", "")) == DEMO_NOTICE
                 else "FIELDED RESPONSES",
    }


def attribute_importance(fit: Dict) -> pd.DataFrame:
    """Relative importance of each attribute: level utility range / total range."""
    coefs = fit["coefficients"]
    ranges = coefs.groupby("attribute")["utility"].agg(lambda s: float(s.max() - s.min()))
    total = float(ranges.sum())
    out = ranges.rename("utility_range").to_frame()
    out["importance"] = out["utility_range"] / total if total else 0.0
    out = out.sort_values("importance", ascending=False).reset_index()
    out["label"] = fit.get("label", "")
    return out


def describe_tasks(tasks: List[ChoiceTask], stage2: Optional[List[ChoiceTask]] = None
                   ) -> pd.DataFrame:
    """Long-form table of every task, profile and attribute level."""
    rows = []
    for stage, task_list in ((1, tasks), (2, stage2 or [])):
        for t in task_list:
            for side, vig in (("A", t.vignette_a), ("B", t.vignette_b)):
                rows.append({"stage": stage, "task_id": t.task_id, "side": side,
                             "profile_id": vig.profile_id,
                             "route_disclosed": vig.route_disclosed, **vig.attributes})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Status, mapping, copy
# ---------------------------------------------------------------------------

def survey_maps_to_ledger() -> pd.DataFrame:
    """Which ledger parameters this instrument would replace with estimates."""
    df = get_commercial_assumptions()
    names = ["mnl_asc_phvs_oral", "mnl_asc_existing_oral", "mnl_price_coef_per_ref",
             "mnl_switch_cost", "mnl_efficacy_bonus", "mnl_asc_current",
             "base_switching_rate"] + [p for p in df["parameter"]
                                       if p.startswith(("oral_pref_", "addressable_"))]
    rows = []
    for n in names:
        match = df[df["parameter"] == n]
        if match.empty:
            rows.append({"parameter": n, "current_value": np.nan,
                         "verification_status": "not_in_ledger",
                         "measurement": "DCE part-worth plus choice-based calibration"})
        else:
            r = match.iloc[0]
            rows.append({"parameter": n, "current_value": r["value"],
                         "verification_status": r["verification_status"],
                         "measurement": "DCE part-worth plus choice-based calibration"})
    out = pd.DataFrame(rows)
    out["status"] = "ASSUMED - survey not conducted"
    return out


def demonstrate_survey_design(n_profiles: int = 32, n_respondents: int = 100,
                              tasks_per_respondent: int = 12, seed: int = 42) -> Dict:
    """Build the design and a clearly labelled synthetic demo dataset."""
    profiles = generate_fractional_factorial(n_profiles=n_profiles, seed=seed)
    for i, p in enumerate(profiles):
        p["profile_id"] = f"P{i:02d}"
    tasks = create_choice_tasks(profiles, n_tasks_per_respondent=tasks_per_respondent,
                                n_respondents=n_respondents, seed=seed)
    assignments = assign_tasks_to_respondents(tasks, n_respondents=n_respondents,
                                              tasks_per_respondent=tasks_per_respondent,
                                              seed=seed)
    demo = simulate_demo_responses(tasks, n_respondents=n_respondents,
                                   tasks_per_respondent=tasks_per_respondent, seed=seed + 1)
    fit = fit_conditional_logit(demo, tasks)
    return {"profiles": profiles, "tasks": tasks,
            "stage2_tasks": create_stage2_reveal(tasks),
            "assignments": assignments, "demo_responses": demo,
            "fit": fit, "importance": attribute_importance(fit),
            "survey_status": survey_status(),
            "notice": DEMO_NOTICE}


def explain_survey_limitations() -> str:
    return (
        "## What the instrument can and cannot settle\n\n"
        "**Status: the survey has not been conducted.** No physician has answered these "
        "questions for PHVS. Everything below describes a design, not data.\n\n"
        "**What a completed DCE would inform:** part-worth utilities for efficacy, route, "
        "dosing frequency, safety experience, out-of-pocket cost and formulary tier; the "
        "oral-preference increment by segment; and how route disclosure in stage 2 moves "
        "choices. Those are exactly the ``mnl_*`` and ``oral_preference_*`` coefficients the "
        "prescribing map currently assumes.\n\n"
        "**What it would still not inform:** actual prescribing. Stated preference in a "
        "hypothetical task is not revealed behaviour; inertia, contracts, PBM leverage, "
        "rebates and patient adherence sit outside the instrument. Real switching rates need "
        "claims data or a launch analogue, and long-term safety emerges only post-marketing.\n\n"
        "**Until real responses exist:** the app shows 'Survey not conducted'. Synthetic "
        "responses may be generated only in visibly labelled demo mode, and fitted utilities "
        "from them are demo output, never evidence."
    )
