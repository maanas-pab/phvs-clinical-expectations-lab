import numpy as np
import pytest

from phvs_lab.modules import response_engine as re_


def test_nested_thresholds_reconstruct_to_sample_size(trials):
    dists = re_.distributions_for_trial("CHAPTER-3", trials)
    for dist in dists.values():
        ok, errors = dist.validate()
        assert ok, errors
        assert sum(dist.counts.values()) == dist.total_patients
        assert all(v >= 0 for v in dist.counts.values())


def test_exclusive_categories_are_subtraction_of_cumulative(trials):
    dist = re_.distributions_for_trial("CHAPTER-3", trials)["Deucrictibant XR 40 mg QD"]
    c = dist.cumulative
    counts = dist.counts
    assert counts["attack-free (100%)"] == c[4]
    assert counts["90-99% reduction"] == c[3] - c[4]
    assert counts["70-89% reduction"] == c[2] - c[3]
    assert counts["50-69% reduction"] == c[1] - c[2]
    assert counts["<50% reduction (non-responder)"] == dist.total_patients - c[1]


def test_non_monotone_thresholds_are_rejected():
    with pytest.raises(ValueError, match="Nested thresholds violated"):
        re_.reconstruct_distribution(55, {1: 20, 2: 40, 3: 50, 4: 54})


def test_count_above_sample_size_is_rejected():
    with pytest.raises(ValueError, match="must lie in \\[0, n\\]"):
        re_.reconstruct_distribution(55, {4: 90})


def test_partial_disclosure_returns_identified_set_not_a_point():
    bounds = re_.category_bounds(100, {1: 60, 3: 30})
    assert (bounds["min"] <= bounds["max"] + 1e-9).all()
    assert not bounds["point_identified"].all()
    assert bounds["min"].sum() <= 100 + 1e-6
    assert bounds["max"].sum() >= 100 - 1e-6
    assert ((bounds["min"] >= 0) & (bounds["max"] <= 100)).all()


def test_full_disclosure_gives_a_single_point():
    bounds = re_.category_bounds(100, {1: 60, 2: 45, 3: 30, 4: 20})
    assert bounds["point_identified"].all()
    assert (bounds["min"] == bounds["max"]).all()


def test_identified_set_contains_the_full_disclosure_answer():
    full = re_.reconstruct_distribution(100, {1: 60, 2: 45, 3: 30, 4: 20})
    bounds = re_.category_bounds(100, {1: 60, 2: 45, 3: 30, 4: 20}).set_index("category")
    for name, value in full.counts.items():
        assert bounds.loc[name, "min"] - 1e-6 <= value <= bounds.loc[name, "max"] + 1e-6


def test_exact_binomial_interval_contains_the_observed_rate():
    lo, hi = re_.clopper_pearson(25, 55)
    assert 0 <= lo <= 25 / 55 <= hi <= 1
    lo0, hi0 = re_.clopper_pearson(0, 55)
    assert lo0 == 0
    l0n, hin = re_.clopper_pearson(55, 55)
    assert hin == 1


def test_trials_without_counts_cannot_be_reconstructed(trials):
    with pytest.raises(ValueError, match="responder counts"):
        re_.distributions_for_trial("HELP-03", trials)


def test_sample_size_resolution_is_one_patient():
    res = re_.sample_size_resolution(55)
    assert res["one_patient_share"] == pytest.approx(1 / 55)
    assert "steps" in res["note"]


def test_attack_free_is_monotone_in_window(trials=None):
    values = [re_.attack_free_probability(0.8, 1.0, w) for w in (30, 90, 168, 365)]
    assert all(a >= b for a, b in zip(values, values[1:]))
