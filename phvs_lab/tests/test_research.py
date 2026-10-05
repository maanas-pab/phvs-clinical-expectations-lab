"""Tests for the research component: recommendation, three RQs, chain, exhibits."""

import pandas as pd
import pytest

from phvs_lab.modules import research as rs


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def test_short_recommendation_leads_with_the_position():
    rec = rs.short_recommendation()
    assert rec["recommendation"] == "Short"
    assert rec["ticker"] == "PHVS"
    assert rec["price"] == pytest.approx(31.19)
    assert rec["price_date"] == "2026-10-02"


def test_short_recommendation_numbers_match_the_verified_models():
    rec = rs.short_recommendation()
    assert rec["model_value"] == pytest.approx(41.136, abs=1e-3)
    assert rec["model_vs_price"] == pytest.approx(
        rec["model_value"] / rec["price"] - 1, rel=1e-9)
    assert rec["required_multiple_of_cashflow_model"] == pytest.approx(4.1193, rel=1e-3)
    assert rec["required_multiple_of_reference"] == pytest.approx(0.7235, rel=1e-3)
    assert rec["breakeven_value_per_share"] == pytest.approx(29.3186, rel=1e-4)
    assert rec["breakeven_value_per_share"] < rec["price"]
    assert rec["bear_value"] < rec["reference_value"] < rec["bull_value"]
    assert rec["spread_multiple"] == pytest.approx(
        rec["bull_value"] / rec["bear_value"], rel=1e-9)


def test_short_recommendation_states_invalidation_not_fair_value():
    rec = rs.short_recommendation()
    thesis = rec["thesis"]
    assert "invalidated" in thesis or "wrong" in thesis
    assert "$" in thesis or "model value" in thesis
    assert len(rec["findings"]) == 5
    assert [f["rank"] for f in rec["findings"]] == [1, 2, 3, 4, 5]
    for f in rec["findings"]:
        assert f["headline"] and f["claim"] and f["basis"]
        assert f["status"] in rs.STATUS


# ---------------------------------------------------------------------------
# Research blocks
# ---------------------------------------------------------------------------

BLOCKS = [rs.rq1_response(), rs.rq2_adoption(), rs.rq3_price()]


@pytest.mark.parametrize("block", BLOCKS, ids=lambda b: b.key)
def test_block_is_resolved_in_four_steps(block):
    assert block.key in {"RQ1", "RQ2", "RQ3"}
    assert block.status in rs.STATUS
    for field in ("question", "methodology", "finding", "implication",
                  "valuation_impact"):
        assert len(getattr(block, field)) > 80, f"{block.key}.{field} too thin"
    assert block.metrics
    assert block.table is not None and not block.table.empty
    assert block.thresholds is not None and not block.thresholds.empty


def test_block_metric_labels_are_unique_and_populated():
    for block in BLOCKS:
        assert len(block.metrics) == len(set(block.metrics))
        for value in block.metrics.values():
            assert isinstance(value, str) and value


def test_rq1_attack_free_is_the_published_count():
    rq1 = rs.rq1_response()
    assert rq1.metrics["Attack-free at 168 days"] == "45.5%"
    assert rq1.metrics["Headline mean reduction"] == "83%"
    assert rq1.metrics["Threshold rows still provisional"] == "3 of 4"


def test_rq2_thresholds_are_proportions_in_range():
    rq2 = rs.rq2_adoption()
    for v in rq2.thresholds["value"]:
        assert 0.0 <= float(v) <= 1.0


def test_rq3_stated_vs_required_are_both_present():
    rq3 = rs.rq3_price()
    assert set(rq3.table["input"]) >= {
        "Peak prophylaxis penetration", "Prophylaxis approval probability",
        "Discount rate", "Annual net price"}
    assert (rq3.table["required_for_price"] != rq3.table["stated"]).any()


def test_status_where_data_cannot_answer_is_threshold_not_conclusion():
    """Every RQ3 break-even row must be a level, not a directional claim."""
    rq3 = rs.rq3_price()
    for note in rq3.thresholds["status"].astype(str):
        assert note in {"model solve", "calculated"}


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

def test_chain_links_exposes_every_connection():
    links = rs.chain_links()
    assert list(links.columns) == ["step", "link", "input", "value", "type",
                                   "status", "reaches_value_directly", "note"]
    assert set(links["step"]) == {1, 2, 3, 4, 5, 6}
    direct = links[links["reaches_value_directly"] == "yes"]
    assert not direct.empty
    assert set(links["status"]) <= {"verified_primary", "corroborated_secondary",
                                    "verified_secondary", "unverified"}


def test_chain_break_even_has_stated_and_threshold():
    be = rs.chain_break_even()
    assert {"stage", "threshold", "at_reference", "break_even",
            "note"} == set(be.columns)
    assert set(be["stage"]) == {"clinical", "adoption", "valuation"}
    assert be["break_even"].astype(str).str.len().min() > 0


def test_chain_curve_is_monotone_in_penetration():
    curve = rs.chain_curve()
    y = pd.Series(curve["value_per_share"])
    assert (y.diff().dropna() > 0).all()
    assert curve["price"] == pytest.approx(31.19)
    assert curve["breakeven"] < curve["price"]


# ---------------------------------------------------------------------------
# Workstreams and disconfirming evidence
# ---------------------------------------------------------------------------

def test_workstreams_distinguish_completed_proposed_assumed():
    ws = rs.workstreams()
    assert set(ws["status"]) == {"completed", "proposed", "assumed"}
    assert (ws["status"] == "completed").sum() >= 5
    for col in ("workstream", "evidence", "used_for"):
        assert ws[col].astype(str).str.len().min() > 5


def test_disconfirming_evidence_is_not_thin():
    df = rs.disconfirming_evidence()
    assert list(df.columns) == ["point", "detail", "why_it_bites"]
    assert len(df) >= 6
    assert df["detail"].str.contains("31.19").any()
    assert df["detail"].str.contains("carry", case=False).any()


def test_limitations_notes_are_shared_and_specific():
    notes = rs.limitations_notes()
    assert len(notes) >= 6
    for topic, body in notes:
        assert isinstance(topic, str) and topic
        assert len(body) > 80


# ---------------------------------------------------------------------------
# Exhibits
# ---------------------------------------------------------------------------

def test_three_exhibits_cover_the_three_research_questions():
    exs = rs.exhibits()
    assert [e["id"] for e in exs] == ["EX1", "EX2", "EX3"]
    assert [e["rq"] for e in exs] == ["RQ1", "RQ2", "RQ3"]
    for e in exs:
        for key in ("png", "title", "headline", "question", "methodology",
                    "caption", "implication", "status", "sources"):
            assert e[key], f"{e['id']}.{key} empty"
        assert len(e["sources"]) >= 3
        for s in e["sources"]:
            assert s["url"].startswith("http")


def test_exhibit_png_names_match_the_export_set():
    assert set(rs.EXHIBIT_PNGS.values()) == {
        "phvs_exhibit_1_response_distribution",
        "phvs_exhibit_2_adoption_constraint",
        "phvs_exhibit_3_evidence_to_downside",
    }
    for e in rs.exhibits():
        assert e["png"] == rs.EXHIBIT_PNGS[e["id"]]


def test_exhibit_source_lines_are_citable():
    for e in rs.exhibits():
        lines = rs.exhibit_source_lines(e)
        assert len(lines) == len(e["sources"])
        assert all(l.startswith("[S") for l in lines)
        assert all("http" in l for l in lines)


def test_exhibit_figures_render_all_three():
    figs = rs.exhibit_figures()
    assert set(figs) == set(rs.EXHIBIT_PNGS.values())
    for name, fig in figs.items():
        assert len(fig.data) >= 2, name
