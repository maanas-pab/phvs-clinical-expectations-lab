from dataclasses import replace

import pytest

from phvs_lab.modules import valuation as val


def test_pitch_reproduction_is_arithmetically_closed():
    rep = val.reproduce_pitch_valuation()
    assert rep["enterprise_value"] == pytest.approx(
        rep["explicit_pv"] + rep["terminal_pv"], rel=1e-6)
    assert rep["equity_value"] == pytest.approx(
        rep["enterprise_value"] + rep["net_cash"], rel=1e-9)
    assert rep["value_per_share_current_shares"] == pytest.approx(
        rep["equity_value"] / rep["shares_current"], rel=1e-9)


def test_pitch_required_equity_is_market_cap_at_the_observed_price(base_params):
    rep = val.reproduce_pitch_valuation()
    price = base_params.stock_price
    assert rep["required_equity_for_price"] == pytest.approx(
        price * rep["shares_current"], rel=1e-9)


def test_pitch_required_multiple_matches_the_ev_ratio():
    rep = val.reproduce_pitch_valuation()
    required_ev = rep["required_equity_for_price"] - rep["net_cash"]
    assert rep["multiple_required"] == pytest.approx(
        required_ev / rep["enterprise_value"], rel=1e-6)
    assert rep["multiple_required"] > 4
    assert rep["enterprise_value"] == pytest.approx(443443975, rel=1e-6)


def test_equity_bridge_identity_holds_for_reference_model(base_params):
    out = val.value_summary(base_params)
    assert out["equity_value"] == pytest.approx(
        out["enterprise_value"] + base_params.net_cash, rel=1e-9)
    assert out["value_per_share"] == pytest.approx(
        out["equity_value"] / base_params.shares, rel=1e-9)


def test_pitch_findings_document_the_share_count_and_terminal_omissions():
    rep = val.reproduce_pitch_valuation()
    text = " ".join(f["check"] + " " + f["message"] for f in rep["findings"]).lower()
    assert "share count" in text and "70.2" in text and "38.0m" in text
    assert "terminal" in text
    assert "discount rate" in text


def test_scenario_ordering_is_respected(base_params):
    table = val.scenario_table(base_params).set_index("scenario")
    assert (table.loc["bear", "value_per_share"]
            <= table.loc["reference", "value_per_share"]
            <= table.loc["bull", "value_per_share"])


def test_market_implied_solutions_reproduce_the_price(base_params):
    price = base_params.stock_price
    impl = val.market_implied(base_params)
    attainable = impl[impl["attainable"]]
    assert len(attainable) >= 3
    for _, row in attainable.iterrows():
        solved = replace(base_params, **{row["input"]: row["required_value"]})
        assert val.value_summary(solved)["value_per_share"] == pytest.approx(
            price, rel=0.03)


def test_required_bundle_is_consistent_with_the_market_cap(base_params):
    bundle = val.required_bundle(base_params)
    price = base_params.stock_price
    assert bundle["required_equity_value"] == pytest.approx(
        price * base_params.shares, rel=1e-9)
    assert bundle["required_enterprise_value"] == pytest.approx(
        bundle["required_equity_value"] - base_params.net_cash, rel=1e-9)


def test_terminal_value_is_a_minority_of_enterprise_value(base_params):
    out = val.value_summary(base_params)
    assert out["pv_terminal"] >= 0
    assert out["terminal_share_of_ev"] < 1


def test_invalidation_conditions_are_populated():
    inv = val.invalidation_conditions()
    assert not inv.empty
    assert {"category", "condition", "why_it_matters", "measured_by"} <= set(inv.columns)
