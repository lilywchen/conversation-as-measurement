# 📏 Conversation as Measurement

🩺 Code accompanying the COLM 2026 paper:

**“Conversation as Measurement in Clinical Encounters: Observable Phase Structure,
Partially Observable Patient State.”**

This repository contains the annotation prompts, evaluation metrics, table code, and
original plotting scripts used in the paper. With three local DataFrames, one command
generates the main PROM table and all five data-driven figures.

> **Data availability:** The clinical transcripts and linked patient-reported outcomes
> cannot be released because they contain protected health information. Bring your own
> approved local inputs; the reproduction command does not modify them.

## 🚀 Quick start

Python 3.10 or 3.11 is supported.

```bash
cd /path/to/conversation-as-measurement
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Run the complete reproduction pipeline:

```bash
conversation-measurement-reproduce \
  --segments "/path/to/segments.csv" \
  --questions "/path/to/questions.csv" \
  --proms "/path/to/proms_df.csv" \
  --output "/path/to/reproduced_outputs"
```

CSV, Parquet, and Feather inputs are supported.

Paper-value verification is enabled by default. When applying the pipeline to a
different dataset, append `--no-verify-published`; the figures and tables will still be
generated, but differing PROM values will not stop the command.

## 📥 Expected inputs

The workflow uses three separate DataFrames:

| Input | Required columns | Used for |
|---|---|---|
| Segments | `doc`, `segment_title`, `minutes_diff`, `segment_start`, `clinician` | Phase allocation, clinician comparison, and sequence clustering |
| Questions | `doc`, `segment_title`, `speaker` | Question distribution and speaker share |
| PROMs | `doc`, `Test`, `Question`, `score`, `true_score` | PROM agreement table and item missingness |

Column meanings:

- `doc` is a stable encounter or visit ID used to group rows. It is not document text and
  is not printed in the outputs.
- Segment rows provide the phase label (`segment_title`), its duration in minutes
  (`minutes_diff`), chronological position (`segment_start`), and grouping value
  (`clinician`). Clinician values are shown only as A–D in the paper figure.
- Question rows identify the phase and whether the question came from the `clinician` or
  `patient`.
- PROM rows identify the instrument (`Test`) and item (`Question`). `score` is the
  transcript-derived 0–4 rating; `true_score` is the patient-reported 0–4 reference.
  Either score may be missing.

Extra columns are ignored. If your column names differ, copy
[`reproduction_config.example.json`](reproduction_config.example.json), edit the paths
and mappings, and run:

```bash
conversation-measurement-reproduce --config /path/to/reproduction_config.json
```

See [REPRODUCE.md](REPRODUCE.md) for accepted values, calculation details, and the
verification rules.

## 📤 Generated outputs

```text
reproduced_outputs/
├── figures/       # Five paper figures as PDFs
├── tables/        # Main PROM summary as CSV and LaTeX
└── verification/  # Cell-by-cell comparison with the paper
```

The command checks the recomputed main PROM table against the values reported in the
paper and identifies any mismatched cells.

## 📊 Paper outputs

The final aggregate outputs reported in the paper are included for reference:

- Phase allocation: [overall](paper_outputs/figures/overall_phase_alloc.pdf) and
  [by clinician group](paper_outputs/figures/clinicians.pdf)
- [Question distribution and speaker share](paper_outputs/figures/question_panel.pdf)
- [Encounter sequence profiles](paper_outputs/figures/sequence.pdf)
- [PROM item missingness](paper_outputs/figures/survey.pdf)
- Main PROM summary table: [CSV](paper_outputs/tables/prom_summary.csv) and
  [LaTeX](paper_outputs/tables/prom_summary.tex)

These files contain aggregate results only. The underlying transcripts and row-level
study data are not included.

## Repository contents

```text
.
├── paper_outputs/                   # Final aggregate paper figures and table
├── prompts/                         # Annotation templates and PROM items
├── src/conversation_measurement/
│   ├── metrics.py                   # Trusted metrics plus paper-specific conventions
│   ├── tables.py                    # Main PROM table
│   ├── paper_figures/               # Original plotting scripts
│   ├── reproduction_inputs.py       # Config, validation, and column adaptation
│   ├── reproduce.py                 # CLI and pipeline orchestration
│   └── verification.py              # Published-result comparison
├── tests/                           # Tests using invented data
├── reproduction_config.example.json
└── REPRODUCE.md                     # Detailed reproduction guide
```

The exact phase/question prompt, PROM-scoring prompt, and PROM item lists are included
in [`prompts/`](prompts/). No model client or inference service is required to reproduce
the tables and figures from existing annotations.

QWK and MAE delegate to scikit-learn's tested implementations. The repository keeps
only the study-specific behavior around them: filtering missing pairs, rounding
half-point patient scores upward for QWK, and defining signed error as patient-reported
minus transcript-derived score.

## 🧪 Tests

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

The tests use invented values and do not contain clinical data.

- `test_metrics.py` checks the metric formulas and missing-value behavior.
- `test_tables.py` checks PROM grouping, missingness, and half-point QWK handling.
- `test_reproduce.py` runs the full three-input pipeline with invented files and checks
  that all five figures, the PROM table, and verification outputs are created.

## 🔒 Privacy

Do not commit transcripts, row-level clinical data, model outputs containing transcript
excerpts, clinician identifiers, credentials, or generated logs. See
[SECURITY.md](SECURITY.md) for reporting instructions.

## Citation

Feel free to cite our paper!

```
@misc{chen2026conversationmeasurementclinicalencounters,
      title={Conversation as Measurement in Clinical Encounters: Observable Phase Structure, Partially Observable Patient State}, 
      author={Lily Chen and Ted Mau and Michael Gensheimer and Brian Anthony Nuyen and Nancy Jiang and James Zou},
      year={2026},
      eprint={2608.08868},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.08868}, 
}
```

## License

This project is released under the [MIT License](LICENSE).
