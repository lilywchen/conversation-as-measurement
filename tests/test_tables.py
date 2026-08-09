import pandas as pd
import pytest

from conversation_measurement.tables import prom_summary_table
from conversation_measurement.verification import PUBLISHED_PROM_SUMMARY, compare_with_published


def test_prom_table_uses_configurable_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "kind": ["A", "A", "B", "B"],
            "truth": [0, 2, 4, 3],
            "inferred": [0, None, 3, 2],
        }
    )
    table = prom_summary_table(
        dataframe,
        reference_col="truth",
        estimate_col="inferred",
        group_col="kind",
        group_labels={"A": "Alpha", "B": "Beta"},
    )
    assert table["Group"].tolist() == ["Overall", "Alpha", "Beta"]
    assert table.loc[0, "N"] == 4
    assert table.loc[0, "Missing (%)"] == pytest.approx(25.0)


def test_prom_table_handles_half_points_and_nullable_values() -> None:
    half_points = pd.DataFrame(
        {"truth": [0.5, 1.5, 2.5, 3.5], "inferred": [1, 2, 3, 4]}
    )
    rounded_table = prom_summary_table(
        half_points,
        reference_col="truth",
        estimate_col="inferred",
        round_reference_half_up_for_qwk=True,
    )
    assert rounded_table.loc[0, "QWK"] == pytest.approx(1.0)
    assert rounded_table.loc[0, "MAE"] == pytest.approx(0.5)

    nullable = pd.DataFrame(
        {
            "kind": ["A", "A"],
            "truth": pd.Series([1, 2], dtype="Int64"),
            "inferred": pd.Series([1, pd.NA], dtype="Int64"),
        }
    )
    nullable_table = prom_summary_table(
        nullable,
        reference_col="truth",
        estimate_col="inferred",
        group_col="kind",
        group_order=["A", "B"],
    )
    assert nullable_table.loc[0, "Missing (%)"] == pytest.approx(50.0)
    assert nullable_table.loc[2, "N"] == 0
    assert pd.isna(nullable_table.loc[2, "Missing (%)"])


def test_published_comparison_matches_displayed_values() -> None:
    displayed = pd.DataFrame(PUBLISHED_PROM_SUMMARY)
    comparison = compare_with_published(displayed)
    assert comparison["Match"].all()
