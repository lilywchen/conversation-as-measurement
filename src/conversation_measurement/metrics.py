"""Evaluation metrics used in the paper.

All functions accept ordinary Python sequences. Missing pairs (``None`` or NaN)
are excluded from agreement and error metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _paired_values(
    reference: Sequence[Any], estimate: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray]:
    if len(reference) != len(estimate):
        raise ValueError("reference and estimate must have the same length")

    pairs = [
        (float(true_value), float(pred_value))
        for true_value, pred_value in zip(reference, estimate, strict=True)
        if not _is_missing(true_value) and not _is_missing(pred_value)
    ]
    if not pairs:
        raise ValueError("no non-missing pairs are available")

    true_values, pred_values = zip(*pairs, strict=True)
    return np.asarray(true_values), np.asarray(pred_values)


def mean_absolute_error(reference: Sequence[Any], estimate: Sequence[Any]) -> float:
    """Return mean absolute error over non-missing pairs."""

    true_values, pred_values = _paired_values(reference, estimate)
    return float(np.mean(np.abs(true_values - pred_values)))


def mean_signed_error(reference: Sequence[Any], estimate: Sequence[Any]) -> float:
    """Return mean ``reference - estimate`` over non-missing pairs.

    With the paper's convention, a positive value means that transcript-derived
    estimates underestimate the corresponding patient-reported scores.
    """

    true_values, pred_values = _paired_values(reference, estimate)
    return float(np.mean(true_values - pred_values))


def quadratic_weighted_kappa(
    reference: Sequence[Any],
    estimate: Sequence[Any],
    *,
    min_rating: int = 0,
    max_rating: int = 4,
) -> float:
    """Compute quadratic weighted Cohen's kappa over non-missing pairs."""

    if max_rating <= min_rating:
        raise ValueError("max_rating must be greater than min_rating")

    true_values, pred_values = _paired_values(reference, estimate)
    if not np.all(true_values == np.round(true_values)) or not np.all(
        pred_values == np.round(pred_values)
    ):
        raise ValueError("quadratic weighted kappa requires integer ratings")

    true_ratings = true_values.astype(int)
    pred_ratings = pred_values.astype(int)
    if (
        np.any(true_ratings < min_rating)
        or np.any(true_ratings > max_rating)
        or np.any(pred_ratings < min_rating)
        or np.any(pred_ratings > max_rating)
    ):
        raise ValueError("ratings fall outside the requested range")

    n_ratings = max_rating - min_rating + 1
    observed = np.zeros((n_ratings, n_ratings), dtype=float)
    for true_rating, pred_rating in zip(true_ratings, pred_ratings, strict=True):
        observed[true_rating - min_rating, pred_rating - min_rating] += 1

    true_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(true_hist, pred_hist) / observed.sum()

    indices = np.arange(n_ratings)
    weights = ((indices[:, None] - indices[None, :]) / (n_ratings - 1)) ** 2
    observed_disagreement = float(np.sum(weights * observed))
    expected_disagreement = float(np.sum(weights * expected))

    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def exact_string_match_rate(
    extracted_strings: Sequence[str],
    source_texts: Sequence[str],
    *,
    lowercase: bool = True,
) -> float:
    """Return the share of extracted strings found verbatim in paired source text.

    This is a string-faithfulness check and is not an estimate of extraction recall.
    """

    if len(extracted_strings) != len(source_texts):
        raise ValueError("extracted_strings and source_texts must have the same length")
    if len(extracted_strings) == 0:
        raise ValueError("at least one extracted string is required")

    matches = 0
    for extracted, source in zip(extracted_strings, source_texts, strict=True):
        if lowercase:
            extracted, source = extracted.lower(), source.lower()
        matches += int(extracted in source)
    return matches / len(extracted_strings)


def prom_summary(
    reference: Sequence[Any],
    estimate: Sequence[Any],
    *,
    qwk_reference: Sequence[Any] | None = None,
) -> dict[str, float | int]:
    """Summarize PROM missingness and agreement using the paper's conventions."""

    if len(reference) != len(estimate):
        raise ValueError("reference and estimate must have the same length")
    if len(reference) == 0:
        raise ValueError("at least one item is required")

    n_missing = sum(_is_missing(value) for value in estimate)
    true_values, pred_values = _paired_values(reference, estimate)
    qwk_values = reference if qwk_reference is None else qwk_reference
    if len(qwk_values) != len(estimate):
        raise ValueError("qwk_reference and estimate must have the same length")
    return {
        "n_total": len(reference),
        "n_missing": n_missing,
        "missing_pct": 100.0 * n_missing / len(reference),
        "n_agreement": len(true_values),
        "qwk": quadratic_weighted_kappa(qwk_values, estimate),
        "mae": mean_absolute_error(true_values, pred_values),
        "mean_signed_error": mean_signed_error(true_values, pred_values),
    }
