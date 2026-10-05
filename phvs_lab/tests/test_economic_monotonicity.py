from dataclasses import replace

import numpy as np
import pytest

from phvs_lab.modules import economics as ec
from phvs_lab.modules import prescribing_map as pm
from phvs_lab.modules import valuation as val


def value_of(base_params, **kw):
    return val.value_summary(replace(base_params, **kw))["value_per_share"]


def test_value_rises_with_peak_penetration(base_params):
    assert (value_of(base_params, peak_penetration_prophylaxis=0.10)
            < value_of(base_params, peak_penetration_prophylaxis=0.15)
            < value_of(base_params, peak_penetration_prophylaxis=0.20))


def test_value_rises_with_approval_probability(base_params):
    assert (value_of(base_params, approval_prophylaxis=0.60)
            < value_of(base_params, approval_prophylaxis=0.88))


def test_value_falls_with_discount_rate(base_params):
    assert (value_of(base_params, discount_rate=0.12)
            < value_of(base_params, discount_rate=0.08))


def test_value_falls_with_share_count(base_params):
    assert (value_of(base_params, shares=80_000_000)
            < value_of(base_params, shares=60_000_000))


def test_value_falls_with_prophylaxis_price(base_params):
    assert (value_of(base_params, price_prophylaxis=500_000)
            > value_of(base_params, price_prophylaxis=350_000))


def test_acute_revenue_is_non_increasing_in_efficacy():
    params = ec.summarize_economics()["params"]
    proj = ec.efficacy_projection(params).sort_values("efficacy")
    rev = proj["acute_revenue"].tolist()
    assert all(a >= b - 1e-6 for a, b in zip(rev, rev[1:]))


def test_map_penetration_rises_with_switching(assumptions):
    peaks = [pm.value_for_inputs(1.0, r)["implied_peak_penetration"]
             for r in (0.05, 0.15, 0.25, 0.35)]
    assert all(a < b for a, b in zip(peaks, peaks[1:]))


def test_map_penetration_is_a_probability():
    out = pm.value_for_inputs(1.0, 0.18)
    assert 0 <= out["implied_peak_penetration"] <= 1
    assert out["implied_patients"] <= 17653 + 1.0
    assert out["value_per_share"] == pytest.approx(38.78, rel=0.02)


def test_choice_probabilities_sum_to_one_by_segment():
    table = pm.choice_probabilities()
    totals = table[["p_current", "p_existing_oral", "p_phvs_oral"]].sum(axis=1)
    assert np.allclose(totals, 1.0)
    assert ((table[["p_current", "p_existing_oral", "p_phvs_oral"]] >= 0)
            & (table[["p_current", "p_existing_oral", "p_phvs_oral"]] <= 1)).all().all()


def test_break_even_switching_brackets_the_price():
    be = pm.break_even_switching(1.0)
    assert be["attainable"] is True
    lo = pm.value_for_inputs(1.0, be["required_switching_rate"] - 0.03)["value_per_share"]
    hi = pm.value_for_inputs(1.0, be["required_switching_rate"] + 0.03)["value_per_share"]
    assert lo < 31.19 < hi


def test_value_surface_is_finite_and_ordered_in_price():
    surf = pm.value_surface(premiums=[0.9, 1.0, 1.1, 1.2],
                            switchings=[0.10, 0.15, 0.20])
    assert np.isfinite(surf["value_per_share"]).all()
    for _, group in surf.groupby("switching_rate"):
        vals = group.sort_values("premium_multiplier")["value_per_share"].to_numpy()
        assert all(a <= b + 1e-6 for a, b in zip(vals, vals[1:]))


def test_lower_premium_means_higher_penetration():
    surf = pm.value_surface(premiums=[0.9, 1.0, 1.1], switchings=[0.15])
    vals = surf.sort_values("premium_multiplier")["implied_peak_penetration"].to_numpy()
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_required_bundle_gap_is_negative_at_reference_inputs(base_params):
    bundle = val.required_bundle(base_params)
    assert bundle["gap_equity"] < 0
    assert bundle["required_multiple_of_stated_ev"] < 1
