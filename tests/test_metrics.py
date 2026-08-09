import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from conversation_measurement.metrics import (
    exact_string_match_rate,
    mean_absolute_error,
    mean_signed_error,
    prom_summary,
    quadratic_weighted_kappa,
)


def test_error_metrics_ignore_missing_pairs() -> None:
    reference = [0, 2, 4, 3]
    estimate = [0, None, 3, np.nan]
    assert mean_absolute_error(reference, estimate) == pytest.approx(0.5)
    assert mean_signed_error(reference, estimate) == pytest.approx(0.5)


def test_qwk_matches_sklearn() -> None:
    reference = [0, 0, 1, 2, 3, 4, 4]
    estimate = [0, 1, 1, 2, 2, 3, 4]
    expected = cohen_kappa_score(reference, estimate, weights="quadratic")
    assert quadratic_weighted_kappa(reference, estimate) == pytest.approx(expected)
    assert quadratic_weighted_kappa(reference, reference) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([2, 2], [2, 2]) == pytest.approx(1.0)


def test_qwk_filters_missing_pairs_before_calling_sklearn() -> None:
    reference = [0, 1, 2, 3, 4, None]
    estimate = [0, 2, 2, 4, 3, 1]
    expected = cohen_kappa_score(reference[:-1], estimate[:-1], weights="quadratic")
    assert quadratic_weighted_kappa(reference, estimate) == pytest.approx(expected)


def test_qwk_rejects_fractional_or_out_of_range_ratings() -> None:
    with pytest.raises(ValueError, match="integer"):
        quadratic_weighted_kappa([0, 1.5], [0, 1])
    with pytest.raises(ValueError, match="outside"):
        quadratic_weighted_kappa([0, 5], [0, 4])


def test_prom_summary_conventions() -> None:
    summary = prom_summary([0, 2, 4, 3], [0, None, 3, 2])
    assert summary["n_total"] == 4
    assert summary["n_missing"] == 1
    assert summary["n_agreement"] == 3
    assert summary["missing_pct"] == pytest.approx(25.0)
    assert summary["mae"] == pytest.approx(2 / 3)
    assert summary["mean_signed_error"] == pytest.approx(2 / 3)
    assert not math.isnan(float(summary["qwk"]))


def test_prom_summary_can_use_a_distinct_qwk_reference() -> None:
    reference = [0.5, 1.5, 2.5, 3.5]
    estimate = [1, 2, 3, 4]
    summary = prom_summary(reference, estimate, qwk_reference=[1, 2, 3, 4])
    assert summary["qwk"] == pytest.approx(1.0)
    assert summary["mae"] == pytest.approx(0.5)
    assert summary["mean_signed_error"] == pytest.approx(-0.5)


def test_exact_string_match_is_faithfulness_not_recall() -> None:
    extracted = ["What brings you in?", "Not present"]
    sources = ["Clinician: WHAT BRINGS YOU IN?", "Patient: I feel better."]
    assert exact_string_match_rate(extracted, sources) == pytest.approx(0.5)
