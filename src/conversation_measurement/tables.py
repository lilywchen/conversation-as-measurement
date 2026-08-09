"""Table construction used by the paper reproduction command."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import prom_summary


def _require_columns(dataframe: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


def prom_summary_table(
    dataframe: pd.DataFrame,
    *,
    reference_col: str,
    estimate_col: str,
    group_col: str | None = None,
    group_order: Sequence[Any] | None = None,
    group_labels: Mapping[Any, str] | None = None,
    include_overall: bool = True,
    round_reference_half_up_for_qwk: bool = False,
) -> pd.DataFrame:
    """Create the paper's PROM missingness and agreement summary table."""

    required = [reference_col, estimate_col] + ([group_col] if group_col else [])
    _require_columns(dataframe, required)
    group_labels = group_labels or {}
    groups: list[tuple[str, pd.DataFrame]] = []
    if include_overall:
        groups.append(("Overall", dataframe))
    if group_col:
        values = list(group_order or dataframe[group_col].drop_duplicates().tolist())
        groups.extend(
            (str(group_labels.get(value, value)), dataframe[dataframe[group_col] == value])
            for value in values
        )

    rows = []
    for label, subset in groups:
        reference_series = subset[reference_col].astype(object)
        estimate_series = subset[estimate_col].astype(object)
        reference = reference_series.where(reference_series.notna(), np.nan).tolist()
        estimate = estimate_series.where(estimate_series.notna(), np.nan).tolist()
        qwk_reference = None
        if round_reference_half_up_for_qwk:
            qwk_reference = [
                value if pd.isna(value) else float(np.floor(float(value) + 0.5))
                for value in reference
            ]
        if subset.empty:
            summary = {
                "n_total": 0,
                "missing_pct": float("nan"),
                "qwk": float("nan"),
                "mae": float("nan"),
                "mean_signed_error": float("nan"),
            }
        else:
            try:
                summary = prom_summary(reference, estimate, qwk_reference=qwk_reference)
            except ValueError as error:
                if "no non-missing pairs" not in str(error):
                    raise
                summary = {
                    "n_total": len(subset),
                    "missing_pct": 100.0 * subset[estimate_col].isna().mean(),
                    "qwk": float("nan"),
                    "mae": float("nan"),
                    "mean_signed_error": float("nan"),
                }
        rows.append(
            {
                "Group": label,
                "N": int(summary["n_total"]),
                "Missing (%)": float(summary["missing_pct"]),
                "QWK": float(summary["qwk"]),
                "MAE": float(summary["mae"]),
                "Mean Signed Error": float(summary["mean_signed_error"]),
            }
        )
    return pd.DataFrame(rows)
