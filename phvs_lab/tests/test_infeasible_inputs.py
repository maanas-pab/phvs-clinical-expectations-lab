from dataclasses import replace

import pytest

from phvs_lab.modules import prescribing_map as pm
from phvs_lab.modules import valuation as val


@pytest.mark.parametrize("kw", [
    dict(discount_rate=0.01),
    dict(terminal_growth=0.12),
    dict(peak_penetration_prophylaxis=1.5),
    dict(peak_penetration_prophylaxis=-0.1),
    dict(approval_prophylaxis=1.2),
    dict(price_prophylaxis=-1.0),
    dict(shares=0),
    dict(time_to_peak_prophylaxis=-2.0),
])
def test_infeasible_valuation_inputs_raise(base_params, kw):
    with pytest.raises(val.ValuationError):
        replace(base_params, **kw).validate()


def test_feasible_reference_inputs_validate(base_params):
    assert base_params.validate() is None


def test_zero_shares_cannot_produce_a_per_share_number(base_params):
    broken = replace(base_params, shares=0)
    with pytest.raises(val.ValuationError):
        val.value_summary(broken)


def test_switching_rate_outside_unit_interval_raises(assumptions):
    with pytest.raises(pm.MapError, match="switching_rate"):
        pm.MapParams(df=assumptions, switching_rate=1.5)


def test_negative_premium_raises(assumptions):
    with pytest.raises(pm.MapError, match="premium_multiplier"):
        pm.MapParams(df=assumptions, premium_multiplier=-1)


def test_addressability_outside_unit_interval_raises(assumptions):
    bad = assumptions.copy()
    bad.loc[bad["parameter"] == "addressable_injection_averse", "value"] = 1.7
    with pytest.raises(pm.MapError, match="Addressability"):
        pm.MapParams(df=bad)


def test_segment_shares_must_sum_to_one(assumptions):
    bad = assumptions.copy()
    bad.loc[bad["parameter"] == "seg_residual_no_demand", "value"] = 0.5
    with pytest.raises(Exception, match="sum to 1"):
        pm.MapParams(df=bad)


def test_cumulative_switch_never_exceeds_one(assumptions):
    params = pm.MapParams(df=assumptions, switching_rate=0.95)
    assert 0 <= params.cumulative_switch <= 1
