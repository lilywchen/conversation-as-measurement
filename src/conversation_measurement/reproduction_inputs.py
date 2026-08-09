"""Load, validate, and adapt private DataFrames for paper reproduction.

This module is the boundary between user-specific column names and the original
plotting scripts. Inputs are read without modification, validated, and renamed
in memory to the canonical names expected by the paper code.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PHASE_LABELS = {
    "opening_rapport": "Opening",
    "history": "History",
    "physical_exam": "Exam",
    "assessment": "Assessment",
    "plan_recommendations": "Plan",
    "education_counseling": "Education",
    "closing": "Closing",
    "non_clinical": "Non-clinical",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "verify_published": True,
    "output_dir": "reproduced_outputs",
    "inputs": {
        "segments": None,
        "questions": None,
        "proms": None,
    },
    "columns": {
        "segments": {
            "encounter": "doc",
            "phase": "segment_title",
            "duration": "minutes_diff",
            "order": "segment_start",
            "clinician": "clinician",
        },
        "questions": {
            "encounter": "doc",
            "phase": "segment_title",
            "speaker": "speaker",
        },
        "proms": {
            "encounter": "doc",
            "instrument": "Test",
            "item": "Question",
            "estimate": "score",
            "reference": "true_score",
        },
    },
    "labels": {
        "prom_groups": {"VHI": "Voice", "CSI": "Cough", "EAT": "Swallowing"},
        "prom_order": ["Voice", "Cough", "Swallowing"],
        "clinician_order": None,
        "cluster_names": {
            "0": "Plan-Dominant",
            "1": "Education-Dominant",
            "2": "History-Dominant",
            "3": "Exam-Dominant",
        },
    },
    "analysis": {
        "sequence_clusters": 4,
        "random_state": 0,
    },
}

CANONICAL_COLUMNS = {
    "segments": {
        "encounter": "doc",
        "phase": "segment_title",
        "duration": "minutes_diff",
        "order": "segment_start",
        "clinician": "clinician",
    },
    "questions": {
        "encounter": "doc",
        "phase": "segment_title",
        "speaker": "speaker",
    },
    "proms": {
        "encounter": "doc",
        "instrument": "Test",
        "item": "Question",
        "estimate": "score",
        "reference": "true_score",
    },
}

REQUIRED_INPUTS = tuple(CANONICAL_COLUMNS)


@dataclass(frozen=True)
class ReproductionInputs:
    """Validated, canonically named inputs plus the merged private config."""

    config: dict[str, Any]
    output_dir: Path
    segments: pd.DataFrame
    questions: pd.DataFrame
    proms: pd.DataFrame
    clinician_order: list[Any]


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _resolve_path(value: str | None, config_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def _load_dataframe(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    selected = list(dict.fromkeys(columns))
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, usecols=selected)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path, columns=selected)
        if suffix == ".feather":
            return pd.read_feather(path, columns=selected)
    except (KeyError, ValueError) as error:
        raise KeyError(f"{path.name} is missing one or more configured columns") from error
    raise ValueError(f"unsupported DataFrame format for {path}; use CSV, Parquet, or Feather")


def _require_complete(dataframe: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = {column: int(dataframe[column].isna().sum()) for column in columns}
    missing = {column: count for column, count in missing.items() if count}
    if missing:
        raise ValueError(f"{name} has missing required values: {missing}")


def _canonical_columns(
    dataframe: pd.DataFrame,
    source_columns: Mapping[str, str],
    canonical_columns: Mapping[str, str],
) -> pd.DataFrame:
    rename = {
        source_columns[meaning]: canonical
        for meaning, canonical in canonical_columns.items()
        if source_columns[meaning] != canonical
    }
    return dataframe.rename(columns=rename).copy()


def _load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.expanduser().resolve()
    user_config = json.loads(config_path.read_text(encoding="utf-8"))
    return _deep_merge(DEFAULT_CONFIG, user_config), config_path.parent


def _input_paths(config: Mapping[str, Any], config_dir: Path) -> dict[str, Path]:
    paths = {
        name: _resolve_path(config["inputs"].get(name), config_dir)
        for name in REQUIRED_INPUTS
    }
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise ValueError(f"config is missing required inputs: {missing}")
    return {name: path for name, path in paths.items() if path is not None}


def _validate_segments(segments: pd.DataFrame, columns: Mapping[str, str]) -> None:
    required = list(columns.values())
    _require_complete(segments, required, "segments")
    duration_column = columns["duration"]
    segments[duration_column] = pd.to_numeric(segments[duration_column], errors="raise")
    durations = segments[duration_column]
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("segment durations must be finite and non-negative")
    totals = durations.groupby(segments[columns["encounter"]]).transform("sum")
    if (totals <= 0).any():
        raise ValueError("each segment document must have positive total duration")


def _validate_questions(questions: pd.DataFrame, columns: Mapping[str, str]) -> None:
    _require_complete(questions, list(columns.values()), "questions")


def _validate_proms(proms: pd.DataFrame, columns: Mapping[str, str]) -> None:
    _require_complete(
        proms,
        [columns["encounter"], columns["instrument"], columns["item"]],
        "proms",
    )
    for column in (columns["estimate"], columns["reference"]):
        proms[column] = pd.to_numeric(proms[column], errors="raise")
        values = proms[column].dropna()
        if not np.isfinite(values).all() or ((values < 0) | (values > 4)).any():
            raise ValueError(f"PROM column {column!r} must contain scores from 0 to 4")
    estimates = proms[columns["estimate"]].dropna()
    if (estimates != np.round(estimates)).any():
        raise ValueError("transcript-derived PROM scores must be integers for QWK")


def _normalize_labels(
    segments: pd.DataFrame,
    questions: pd.DataFrame,
    proms: pd.DataFrame,
    columns: Mapping[str, Mapping[str, str]],
    labels: Mapping[str, Any],
) -> None:
    segment_phase = columns["segments"]["phase"]
    question_phase = columns["questions"]["phase"]
    speaker = columns["questions"]["speaker"]
    instrument = columns["proms"]["instrument"]

    segments[segment_phase] = segments[segment_phase].replace(PHASE_LABELS)
    questions[question_phase] = questions[question_phase].replace(PHASE_LABELS)
    questions[speaker] = questions[speaker].astype(str).str.lower()
    proms[instrument] = proms[instrument].replace(labels["prom_groups"])

    valid_phases = set(PHASE_LABELS.values())
    unknown_segments = set(segments[segment_phase]) - valid_phases
    unknown_questions = set(questions[question_phase]) - valid_phases
    if unknown_segments:
        raise ValueError(f"segments contains unknown phase labels: {unknown_segments}")
    if unknown_questions:
        raise ValueError(f"questions contains unknown phase labels: {unknown_questions}")
    if set(questions[speaker]) != {"clinician", "patient"}:
        raise ValueError("questions speaker values must include only clinician and patient")
    prom_groups = set(proms[instrument])
    if prom_groups != set(labels["prom_order"]):
        raise ValueError(
            f"PROM instrument values do not match the configured groups: {prom_groups}"
        )


def load_reproduction_inputs(config_path: Path) -> ReproductionInputs:
    """Load a private config and return validated, canonically named DataFrames."""

    config, config_dir = _load_config(config_path)
    paths = _input_paths(config, config_dir)
    output_dir = _resolve_path(config["output_dir"], config_dir)
    if output_dir is None:
        raise ValueError("output_dir must be set")

    columns = config["columns"]
    frames = {
        name: _load_dataframe(paths[name], list(columns[name].values()))
        for name in REQUIRED_INPUTS
    }
    segments = frames["segments"]
    questions = frames["questions"]
    proms = frames["proms"]

    _validate_segments(segments, columns["segments"])
    _validate_questions(questions, columns["questions"])
    _validate_proms(proms, columns["proms"])
    _normalize_labels(segments, questions, proms, columns, config["labels"])

    canonical_segments = _canonical_columns(
        segments, columns["segments"], CANONICAL_COLUMNS["segments"]
    )
    canonical_questions = _canonical_columns(
        questions, columns["questions"], CANONICAL_COLUMNS["questions"]
    )
    canonical_proms = _canonical_columns(
        proms, columns["proms"], CANONICAL_COLUMNS["proms"]
    )
    canonical_segments["phase_pct"] = canonical_segments["minutes_diff"] / (
        canonical_segments.groupby("doc")["minutes_diff"].transform("sum")
    )

    clinician_values = canonical_segments["clinician"].drop_duplicates().tolist()
    clinician_order = config["labels"]["clinician_order"] or clinician_values
    if len(clinician_order) != len(set(clinician_order)) or set(clinician_order) != set(
        clinician_values
    ):
        raise ValueError("clinician_order must list every clinician group exactly once")
    if int(config["analysis"]["sequence_clusters"]) > canonical_segments["doc"].nunique():
        raise ValueError("sequence_clusters cannot exceed the number of documents")

    return ReproductionInputs(
        config=config,
        output_dir=output_dir,
        segments=canonical_segments,
        questions=canonical_questions,
        proms=canonical_proms,
        clinician_order=clinician_order,
    )
