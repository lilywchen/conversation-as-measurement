import json
import sys

import pandas as pd

import conversation_measurement.reproduce as reproduce_module
from conversation_measurement.reproduce import reproduce


def test_one_command_reproduction_uses_multiple_dataframes(tmp_path) -> None:
    segments = pd.DataFrame(
        {
            "doc": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "segment_title": [
                "history",
                "plan_recommendations",
                "physical_exam",
                "history",
                "education_counseling",
                "plan_recommendations",
                "plan_recommendations",
                "assessment",
            ],
            "minutes_diff": [9, 1, 9, 1, 9, 1, 9, 1],
            "segment_start": [0, 1, 0, 1, 0, 1, 0, 1],
            "clinician": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"],
        }
    )
    questions = pd.DataFrame(
        {
            "doc": ["a", "b", "c", "d"],
            "segment_title": [
                "history",
                "physical_exam",
                "education_counseling",
                "plan_recommendations",
            ],
            "speaker": ["clinician", "patient", "clinician", "patient"],
            "text": ["Question A?", "Question B?", "Question C?", "Question D?"],
            "raw_text": [
                "Question A? Answer.",
                "Question B? Answer.",
                "Question C? Answer.",
                "Question D? Answer.",
            ],
        }
    )
    proms = pd.DataFrame(
        {
            "doc": ["a", "a", "b", "b", "c", "c"],
            "Test": ["VHI", "VHI", "CSI", "CSI", "EAT", "EAT"],
            "Question": ["A", "B", "A", "B", "A", "B"],
            "score": [1, None, 2, 3, 1, 4],
            "true_score": [0.5, 2, 2, 4, 1, 3.5],
        }
    )
    frames = {
        "segments": segments.rename(
            columns={
                "doc": "visit_id",
                "segment_title": "phase_name",
                "minutes_diff": "duration_minutes",
                "segment_start": "phase_start",
                "clinician": "provider_group",
            }
        ),
        "questions": questions.rename(
            columns={
                "doc": "visit_id",
                "segment_title": "phase_name",
                "speaker": "role",
            }
        ),
        "proms": proms.rename(
            columns={
                "doc": "visit_id",
                "Test": "instrument",
                "Question": "item_text",
                "score": "estimate",
                "true_score": "reference",
            }
        ),
    }
    input_paths = {}
    for name, dataframe in frames.items():
        path = tmp_path / f"{name}.csv"
        dataframe.to_csv(path, index=False)
        input_paths[name] = str(path)

    output_dir = tmp_path / "outputs"
    config = {
        "verify_published": False,
        "output_dir": str(output_dir),
        "inputs": input_paths,
        "columns": {
            "segments": {
                "encounter": "visit_id",
                "phase": "phase_name",
                "duration": "duration_minutes",
                "order": "phase_start",
                "clinician": "provider_group",
            },
            "questions": {
                "encounter": "visit_id",
                "phase": "phase_name",
                "speaker": "role",
            },
            "proms": {
                "encounter": "visit_id",
                "instrument": "instrument",
                "item": "item_text",
                "estimate": "estimate",
                "reference": "reference",
            },
        },
    }
    config_path = tmp_path / "reproduction_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary = reproduce(config_path)

    assert summary["counts"] == {"transcripts": 4, "questions": 4, "prom_items": 6}
    assert (output_dir / "figures" / "overall_phase_alloc.pdf").exists()
    assert (output_dir / "figures" / "clinicians.pdf").exists()
    assert (output_dir / "figures" / "question_panel.pdf").exists()
    assert (output_dir / "figures" / "sequence.pdf").exists()
    assert (output_dir / "figures" / "survey.pdf").exists()
    assert (output_dir / "tables" / "prom_summary.tex").exists()
    assert (output_dir / "verification" / "reproduction_summary.json").exists()


def test_direct_cli_can_skip_published_verification(
    monkeypatch, tmp_path, capsys
) -> None:
    observed_config = {}

    def fake_reproduce(config_path):
        observed_config.update(json.loads(config_path.read_text(encoding="utf-8")))
        return {"completed": True}

    monkeypatch.setattr(reproduce_module, "reproduce", fake_reproduce)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "conversation-measurement-reproduce",
            "--segments",
            str(tmp_path / "segments.csv"),
            "--questions",
            str(tmp_path / "questions.csv"),
            "--proms",
            str(tmp_path / "proms.csv"),
            "--output",
            str(tmp_path / "outputs"),
            "--no-verify-published",
        ],
    )

    reproduce_module.main()

    assert observed_config["verify_published"] is False
    assert json.loads(capsys.readouterr().out) == {"completed": True}
