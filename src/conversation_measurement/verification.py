"""Compare private-data recomputations with aggregate values printed in the paper."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

PUBLISHED_PROM_SUMMARY = (
    {
        "Group": "Overall",
        "N": 2730,
        "Missing (%)": 63.0,
        "QWK": 0.508,
        "MAE": 1.006,
        "Mean Signed Error": 0.391,
    },
    {
        "Group": "Voice",
        "N": 1660,
        "Missing (%)": 65.7,
        "QWK": 0.488,
        "MAE": 1.040,
        "Mean Signed Error": 0.430,
    },
    {
        "Group": "Cough",
        "N": 480,
        "Missing (%)": 67.1,
        "QWK": 0.527,
        "MAE": 0.987,
        "Mean Signed Error": 0.696,
    },
    {
        "Group": "Swallowing",
        "N": 590,
        "Missing (%)": 52.0,
        "QWK": 0.539,
        "MAE": 0.947,
        "Mean Signed Error": 0.134,
    },
)

DISPLAY_DECIMALS = {
    "Missing (%)": 1,
    "Supported (%)": 1,
    "QWK": 3,
    "MAE": 3,
    "Mean Signed Error": 3,
}


def compare_with_published(
    actual: pd.DataFrame,
    *,
    published: str | tuple[Mapping[str, object], ...] = "full",
) -> pd.DataFrame:
    """Compare a recomputed table to displayed paper values after paper rounding."""

    if isinstance(published, str):
        if published != "full":
            raise ValueError("published must be 'full' or explicit rows")
        expected_rows = PUBLISHED_PROM_SUMMARY
    else:
        expected_rows = published

    expected = pd.DataFrame(expected_rows).set_index("Group")
    observed = actual.set_index("Group")
    rows = []
    for group in expected.index:
        if group not in observed.index:
            rows.append(
                {
                    "Group": group,
                    "Column": "Group",
                    "Actual": None,
                    "Expected": group,
                    "Match": False,
                }
            )
            continue
        for column in expected.columns:
            expected_value = expected.at[group, column]
            actual_value = observed.at[group, column]
            decimals = DISPLAY_DECIMALS.get(column, 0)
            matches = round(float(actual_value), decimals) == round(
                float(expected_value), decimals
            )
            rows.append(
                {
                    "Group": group,
                    "Column": column,
                    "Actual": actual_value,
                    "Expected": expected_value,
                    "Match": matches,
                }
            )
    return pd.DataFrame(rows)
