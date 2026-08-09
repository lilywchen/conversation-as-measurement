"""One-command reproduction of aggregate paper tables and data-driven figures."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")
import matplotlib.pyplot as plt  # noqa: E402

from .paper_figures import (
    build_sequence_df_long,
    cluster_docs_by_duration,
    make_question_phase_panel,
    make_seg_bars,
    plot_all_clinicians,
    plot_prom_item_missingness_matched,
    plot_sequence_index,
)
from .tables import prom_summary_table
from .verification import compare_with_published

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

REQUIRED_INPUTS = ("segments", "questions", "proms")


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
    raise ValueError(
        f"unsupported DataFrame format for {path}; "
        "use CSV, Parquet, or Feather"
    )


def _require_columns(dataframe: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _require_complete(dataframe: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = {column: int(dataframe[column].isna().sum()) for column in columns}
    missing = {column: count for column, count in missing.items() if count}
    if missing:
        raise ValueError(f"{name} has missing required values: {missing}")


def _display_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    formats = {
        "Missing (%)": "{:.1f}",
        "QWK": "{:.3f}",
        "MAE": "{:.3f}",
        "Mean Signed Error": "{:.3f}",
    }
    for column, formatter in formats.items():
        if column in display:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else formatter.format(float(value))
            )
    if "N" in display:
        display["N"] = display["N"].map(lambda value: str(int(value)))
    return display


def _write_table(table: pd.DataFrame, stem: Path) -> None:
    table.to_csv(stem.with_suffix(".csv"), index=False)
    latex = _display_table(table).to_latex(index=False)
    stem.with_suffix(".tex").write_text(latex, encoding="utf-8")


def _canonical_columns(
    dataframe: pd.DataFrame, source_columns: Mapping[str, str]
) -> pd.DataFrame:
    rename = {source: target for target, source in source_columns.items() if source != target}
    return dataframe.rename(columns=rename).copy()


def reproduce(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    user_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = _deep_merge(DEFAULT_CONFIG, user_config)
    config_dir = config_path.parent
    input_paths = {
        name: _resolve_path(value, config_dir) for name, value in config["inputs"].items()
    }

    missing_inputs = [name for name in REQUIRED_INPUTS if input_paths[name] is None]
    if missing_inputs:
        raise ValueError(f"config is missing required inputs: {missing_inputs}")
    output_value = config["output_dir"]
    output_dir = _resolve_path(output_value, config_dir)
    if output_dir is None:
        raise ValueError("output_dir must be set")
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    verification_dir = output_dir / "verification"
    for directory in (figures_dir, tables_dir, verification_dir):
        directory.mkdir(parents=True, exist_ok=True)

    columns = config["columns"]
    labels = config["labels"]
    analysis = config["analysis"]
    segment_columns = columns["segments"]
    question_columns = columns["questions"]
    prom_columns = columns["proms"]

    required_columns = {
        "segments": [
            segment_columns["encounter"],
            segment_columns["phase"],
            segment_columns["duration"],
            segment_columns["order"],
            segment_columns["clinician"],
        ],
        "questions": [
            question_columns["encounter"],
            question_columns["phase"],
            question_columns["speaker"],
        ],
        "proms": [
            prom_columns["encounter"],
            prom_columns["instrument"],
            prom_columns["item"],
            prom_columns["estimate"],
            prom_columns["reference"],
        ],
    }
    frames = {
        name: _load_dataframe(input_paths[name], required_columns[name])
        for name in REQUIRED_INPUTS
    }
    segments = frames["segments"].copy()
    questions = frames["questions"].copy()
    proms = frames["proms"].copy()

    _require_columns(
        segments,
        required_columns["segments"],
        "segments",
    )
    _require_columns(
        questions,
        required_columns["questions"],
        "questions",
    )
    _require_columns(
        proms,
        required_columns["proms"],
        "proms",
    )

    _require_complete(segments, required_columns["segments"], "segments")
    _require_complete(questions, required_columns["questions"], "questions")
    _require_complete(
        proms,
        [
            prom_columns["encounter"],
            prom_columns["instrument"],
            prom_columns["item"],
        ],
        "proms",
    )

    segments[segment_columns["duration"]] = pd.to_numeric(
        segments[segment_columns["duration"]], errors="raise"
    )
    durations = segments[segment_columns["duration"]]
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("segment durations must be finite and non-negative")
    document_durations = durations.groupby(segments[segment_columns["encounter"]]).transform(
        "sum"
    )
    if (document_durations <= 0).any():
        raise ValueError("each segment document must have positive total duration")

    for column in (prom_columns["estimate"], prom_columns["reference"]):
        proms[column] = pd.to_numeric(proms[column], errors="raise")
        values = proms[column].dropna()
        if not np.isfinite(values).all() or ((values < 0) | (values > 4)).any():
            raise ValueError(f"PROM column {column!r} must contain scores from 0 to 4")
    estimate_values = proms[prom_columns["estimate"]].dropna()
    if (estimate_values != np.round(estimate_values)).any():
        raise ValueError("transcript-derived PROM scores must be integers for QWK")

    segments[segment_columns["phase"]] = segments[segment_columns["phase"]].replace(
        PHASE_LABELS
    )
    questions[question_columns["phase"]] = questions[question_columns["phase"]].replace(
        PHASE_LABELS
    )
    questions[question_columns["speaker"]] = (
        questions[question_columns["speaker"]].astype(str).str.lower()
    )
    proms[prom_columns["instrument"]] = proms[prom_columns["instrument"]].replace(
        labels["prom_groups"]
    )

    valid_phases = set(PHASE_LABELS.values())
    unknown_segment_phases = set(segments[segment_columns["phase"]]) - valid_phases
    unknown_question_phases = set(questions[question_columns["phase"]]) - valid_phases
    if unknown_segment_phases:
        raise ValueError(f"segments contains unknown phase labels: {unknown_segment_phases}")
    if unknown_question_phases:
        raise ValueError(f"questions contains unknown phase labels: {unknown_question_phases}")
    speakers = set(questions[question_columns["speaker"]])
    if speakers != {"clinician", "patient"}:
        raise ValueError("questions speaker values must include only clinician and patient")
    prom_groups = set(proms[prom_columns["instrument"]])
    if prom_groups != set(labels["prom_order"]):
        raise ValueError(
            "PROM instrument values do not match the configured groups: "
            f"{prom_groups}"
        )

    paper_segments = _canonical_columns(
        segments,
        {
            "doc": segment_columns["encounter"],
            "segment_title": segment_columns["phase"],
            "minutes_diff": segment_columns["duration"],
            "segment_start": segment_columns["order"],
            "clinician": segment_columns["clinician"],
        },
    )
    paper_segments["phase_pct"] = paper_segments["minutes_diff"] / paper_segments.groupby(
        "doc"
    )["minutes_diff"].transform("sum")
    paper_questions = _canonical_columns(
        questions,
        {
            "doc": question_columns["encounter"],
            "segment_title": question_columns["phase"],
            "speaker": question_columns["speaker"],
        },
    )
    paper_proms = _canonical_columns(
        proms,
        {
            "doc": prom_columns["encounter"],
            "Test": prom_columns["instrument"],
            "Question": prom_columns["item"],
            "score": prom_columns["estimate"],
        },
    )
    clinician_order = labels["clinician_order"] or paper_segments[
        "clinician"
    ].dropna().drop_duplicates().tolist()
    clinician_values = paper_segments["clinician"].dropna().drop_duplicates().tolist()
    if len(clinician_order) != len(set(clinician_order)) or set(clinician_order) != set(
        clinician_values
    ):
        raise ValueError("clinician_order must list every clinician group exactly once")
    if int(analysis["sequence_clusters"]) > paper_segments["doc"].nunique():
        raise ValueError("sequence_clusters cannot exceed the number of documents")

    make_seg_bars(paper_segments, filename=figures_dir / "overall_phase_alloc.pdf")
    plt.close("all")
    plot_all_clinicians(
        paper_segments,
        clinicians=clinician_order,
        anon=True,
        savepath=figures_dir / "clinicians.pdf",
    )
    plt.close("all")
    make_question_phase_panel(
        paper_questions,
        filename=figures_dir / "question_panel.pdf",
    )
    plt.close("all")
    plot_prom_item_missingness_matched(
        paper_proms,
        tests=tuple(labels["prom_order"]),
        filename=figures_dir / "survey.pdf",
    )
    plt.close("all")

    sequence_rows = build_sequence_df_long(paper_segments, order_col="segment_start")
    cluster_metadata = cluster_docs_by_duration(
        sequence_rows,
        k_clusters=int(analysis["sequence_clusters"]),
        random_state=int(analysis["random_state"]),
    )
    cluster_names = {int(key): value for key, value in labels["cluster_names"].items()}
    plot_sequence_index(
        sequence_rows,
        cluster_meta=cluster_metadata,
        cluster_labels=cluster_names,
        savepath=figures_dir / "sequence.pdf",
    )
    plt.close("all")

    prom_table = prom_summary_table(
        proms,
        reference_col=prom_columns["reference"],
        estimate_col=prom_columns["estimate"],
        group_col=prom_columns["instrument"],
        group_order=labels["prom_order"],
        round_reference_half_up_for_qwk=True,
    )
    _write_table(prom_table, tables_dir / "prom_summary")
    prom_comparison = compare_with_published(prom_table, published="full")
    prom_comparison.to_csv(verification_dir / "prom_summary_vs_paper.csv", index=False)

    summary: dict[str, Any] = {
        "counts": {
            "transcripts": int(segments[segment_columns["encounter"]].nunique()),
            "questions": int(len(questions)),
            "prom_items": int(len(proms)),
        },
        "generated_figures": sorted(path.name for path in figures_dir.glob("*.pdf")),
        "generated_tables": ["prom_summary"],
        "prom_summary_matches_paper": bool(prom_comparison["Match"].all()),
        "details": {
            "qwk_reference_rule": "round half-point patient scores upward; QWK only",
            "mae_reference_rule": "use original patient-reported scores",
            "signed_error": "patient-reported minus transcript-derived",
            "sequence_clustering": {
                "k": int(analysis["sequence_clusters"]),
                "random_state": int(analysis["random_state"]),
                "n_init": "auto",
            },
        },
    }
    (verification_dir / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if config["verify_published"] and not summary["prom_summary_matches_paper"]:
        raise AssertionError(
            "the recomputed PROM table does not match the paper; "
            f"inspect {verification_dir}"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce aggregate paper tables and data-driven figures "
            "from private DataFrames."
        )
    )
    parser.add_argument("--config", type=Path, help="Path to private JSON config")
    parser.add_argument("--segments", type=Path, help="Segments DataFrame file")
    parser.add_argument("--questions", type=Path, help="Questions DataFrame file")
    parser.add_argument("--proms", type=Path, help="PROM DataFrame file")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument(
        "--no-verify-published",
        action="store_true",
        help="Generate outputs without requiring the PROM table to match the paper",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    direct_values = (args.segments, args.questions, args.proms, args.output)
    if args.config and (
        any(value is not None for value in direct_values) or args.no_verify_published
    ):
        raise SystemExit("use either --config or direct DataFrame flags, not both")
    if args.config:
        summary = reproduce(args.config)
    else:
        missing = [
            name
            for name, value in zip(
                ("--segments", "--questions", "--proms", "--output"),
                direct_values,
                strict=True,
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "direct reproduction requires " + ", ".join(missing) + "; or use --config"
            )
        direct_config = {
            "verify_published": not args.no_verify_published,
            "output_dir": str(args.output.resolve()),
            "inputs": {
                "segments": str(args.segments.resolve()),
                "questions": str(args.questions.resolve()),
                "proms": str(args.proms.resolve()),
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "reproduction_config.json"
            config_path.write_text(json.dumps(direct_config), encoding="utf-8")
            summary = reproduce(config_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
