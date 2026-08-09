"""One-command reproduction of aggregate paper tables and data-driven figures."""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import matplotlib
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
from .reproduction_inputs import ReproductionInputs, load_reproduction_inputs
from .tables import prom_summary_table
from .verification import compare_with_published


def _output_directories(output_dir: Path) -> tuple[Path, Path, Path]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    verification_dir = output_dir / "verification"
    for directory in (figures_dir, tables_dir, verification_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir, verification_dir


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
    stem.with_suffix(".tex").write_text(
        _display_table(table).to_latex(index=False), encoding="utf-8"
    )


def _generate_figures(inputs: ReproductionInputs, figures_dir: Path) -> list[str]:
    """Run the original plotting functions on canonically named DataFrames."""

    make_seg_bars(inputs.segments, filename=figures_dir / "overall_phase_alloc.pdf")
    plt.close("all")
    plot_all_clinicians(
        inputs.segments,
        clinicians=inputs.clinician_order,
        anon=True,
        savepath=figures_dir / "clinicians.pdf",
    )
    plt.close("all")
    make_question_phase_panel(
        inputs.questions,
        filename=figures_dir / "question_panel.pdf",
    )
    plt.close("all")
    plot_prom_item_missingness_matched(
        inputs.proms,
        tests=tuple(inputs.config["labels"]["prom_order"]),
        filename=figures_dir / "survey.pdf",
    )
    plt.close("all")

    sequence_rows = build_sequence_df_long(inputs.segments, order_col="segment_start")
    analysis = inputs.config["analysis"]
    cluster_metadata = cluster_docs_by_duration(
        sequence_rows,
        k_clusters=int(analysis["sequence_clusters"]),
        random_state=int(analysis["random_state"]),
    )
    cluster_names = {
        int(key): value for key, value in inputs.config["labels"]["cluster_names"].items()
    }
    plot_sequence_index(
        sequence_rows,
        cluster_meta=cluster_metadata,
        cluster_labels=cluster_names,
        savepath=figures_dir / "sequence.pdf",
    )
    plt.close("all")
    return sorted(path.name for path in figures_dir.glob("*.pdf"))


def _generate_prom_table(
    inputs: ReproductionInputs, tables_dir: Path, verification_dir: Path
) -> bool:
    prom_table = prom_summary_table(
        inputs.proms,
        reference_col="true_score",
        estimate_col="score",
        group_col="Test",
        group_order=inputs.config["labels"]["prom_order"],
        round_reference_half_up_for_qwk=True,
    )
    _write_table(prom_table, tables_dir / "prom_summary")
    comparison = compare_with_published(prom_table, published="full")
    comparison.to_csv(verification_dir / "prom_summary_vs_paper.csv", index=False)
    return bool(comparison["Match"].all())


def _summary(
    inputs: ReproductionInputs,
    generated_figures: list[str],
    prom_summary_matches_paper: bool,
) -> dict[str, Any]:
    analysis = inputs.config["analysis"]
    return {
        "counts": {
            "transcripts": int(inputs.segments["doc"].nunique()),
            "questions": int(len(inputs.questions)),
            "prom_items": int(len(inputs.proms)),
        },
        "generated_figures": generated_figures,
        "generated_tables": ["prom_summary"],
        "prom_summary_matches_paper": prom_summary_matches_paper,
        "details": {
            "qwk_implementation": "sklearn.metrics.cohen_kappa_score",
            "qwk_reference_rule": "round half-point patient scores upward; QWK only",
            "mae_implementation": "sklearn.metrics.mean_absolute_error",
            "mae_reference_rule": "use original patient-reported scores",
            "signed_error": "patient-reported minus transcript-derived",
            "sequence_clustering": {
                "k": int(analysis["sequence_clusters"]),
                "random_state": int(analysis["random_state"]),
                "n_init": "auto",
            },
        },
    }


def reproduce(config_path: Path) -> dict[str, Any]:
    """Generate the paper outputs described by a private reproduction config."""

    inputs = load_reproduction_inputs(config_path)
    figures_dir, tables_dir, verification_dir = _output_directories(inputs.output_dir)
    generated_figures = _generate_figures(inputs, figures_dir)
    prom_matches = _generate_prom_table(inputs, tables_dir, verification_dir)
    summary = _summary(inputs, generated_figures, prom_matches)
    (verification_dir / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if inputs.config["verify_published"] and not prom_matches:
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
