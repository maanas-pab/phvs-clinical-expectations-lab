import pytest

from phvs_lab.modules import economics as ec


def test_funnel_stages_are_monotone_non_increasing():
    ledger = ec.population_ledger()
    funnel = ledger[~ledger["stage"].str.startswith("Segment:")].reset_index(drop=True)
    assert list(funnel["stage"]) == sorted(funnel["stage"], key=list(funnel["stage"]).index)
    values = funnel["patients"].tolist()
    for a, b in zip(values, values[1:]):
        assert b <= a + 1e-6


def test_segments_are_mutually_exclusive_and_exhaustive():
    shares = ec.segment_shares()
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0 for v in shares.values())


def test_segment_cohorts_reconcile_to_eligible_population():
    ledger = ec.population_ledger()
    eligible = float(ledger.loc[ledger["stage"] == "Clinically eligible for prophylaxis",
                                "patients"].iloc[0])
    segs = ledger[ledger["stage"].str.startswith("Segment:")]
    assert segs["patients"].sum() == pytest.approx(eligible, rel=1e-6)
    assert abs(eligible - 17652.6) < 1.0


def test_segments_fit_inside_the_diagnosed_population():
    ledger = ec.population_ledger()
    diagnosed = float(ledger.loc[ledger["stage"] == "Global diagnosed HAE", "patients"].iloc[0])
    segs = ledger[ledger["stage"].str.startswith("Segment:")]
    assert segs["patients"].sum() <= diagnosed


def test_double_counting_is_detected_and_rejected():
    ledger = ec.population_ledger()
    cohorts = {"cohort_a": 15000.0, "cohort_b": 15000.0}
    with pytest.raises(Exception, match="double|exceed|outside|population"):
        ec.assert_no_double_counting(ledger, cohorts, 20000.0)


def test_negative_population_is_rejected():
    with pytest.raises(Exception):
        ec.assert_no_double_counting(ec.population_ledger(), {}, -10.0)


def test_disjoint_cohorts_pass_validation():
    ledger = ec.population_ledger()
    ec.assert_no_double_counting(ledger, {"a": 500.0, "b": 700.0}, 17652.6)


def test_prophylaxis_patients_counted_once_across_segments():
    shares = ec.segment_shares()
    eligible = 17652.6
    counts = {k: v * eligible for k, v in shares.items()}
    assert sum(counts.values()) == pytest.approx(eligible, rel=1e-9)
    assert len(counts) == 6


def test_bridge_patients_are_inside_the_eligible_population():
    comps = ec.bridge_components(ec.summarize_economics()["params"])
    assert comps["patients"] <= 17653 + 1.0
    assert 0 <= comps["breakthrough_patients"] <= comps["patients"] + 1e-6


def test_more_efficacy_never_creates_more_breakthrough_cases():
    comps = ec.summarize_economics()["params"]
    proj = ec.efficacy_projection(comps)
    rows = proj.sort_values("efficacy")["breakthrough_patients"].tolist()
    assert all(a >= b - 1e-6 for a, b in zip(rows, rows[1:]))


def test_dispersion_calibration_is_reported_and_labelled():
    comps = ec.bridge_components(ec.summarize_economics()["params"])
    assert comps["fitted_dispersion_k"] > 0
    assert "MODEL-DERIVED" in comps["label"]
