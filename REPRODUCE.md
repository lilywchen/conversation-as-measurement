# One-command reproduction

The public reproduction path uses three separate DataFrames. They remain in the approved
private environment and are never copied into this repository.

| Input | Purpose | Required columns with default names |
|---|---|---|
| Segments | Phase allocation, clinician comparison, sequence clustering | `doc`, `segment_title`, `minutes_diff`, `segment_start`, `clinician` |
| Questions | Question distribution and speaker-share figure | `doc`, `segment_title`, `speaker` |
| PROMs | Main missingness/agreement table and item-missingness figure | `doc`, `Test`, `Question`, `score`, `true_score` |

These are semantic requirements, not fixed public names. If your columns have different
names, map them in the config file described below.

## Expected values

| Field | Expected meaning and values |
|---|---|
| `doc` | Stable encounter identifier. It only needs to be consistent within and across the relevant files; it is never printed in an output. |
| `segment_title` | One of the eight internal phase codes (`opening_rapport`, `history`, `physical_exam`, `assessment`, `plan_recommendations`, `education_counseling`, `closing`, `non_clinical`) or the corresponding display label (`Opening`, `History`, `Exam`, `Assessment`, `Plan`, `Education`, `Closing`, `Non-clinical`). |
| `minutes_diff` | Numeric segment duration in minutes, zero or greater. Every encounter must have positive total duration. |
| `segment_start` | Any sortable value that places segments in chronological order within an encounter. |
| `clinician` | Clinician/group identifier. Exactly four values are expected for the paper figure; outputs replace them with A–D. Do not put private identifiers in the public config. |
| `speaker` | `clinician` or `patient` (capitalization is normalized). |
| `Test` | PROM instrument: `VHI`, `CSI`, or `EAT` (or values mapped through the config labels). |
| `Question` | PROM item text or item identifier, consistent within each instrument. |
| `score` | Transcript-derived item score: integer 0–4, or missing when the item is not observable. |
| `true_score` | Patient-reported reference item score from 0–4, including possible half-points, or missing. |

No transcript text, clinician metadata, validation-only DataFrames, or precomputed
percentages are required. A formal schema file is intentionally omitted: the command
validates the required columns and values when it loads the three inputs.

## Direct command

For the versions used to verify this release, install with:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Then run:

```bash
conversation-measurement-reproduce \
  --segments "/PRIVATE/PATH/segments.csv" \
  --questions "/PRIVATE/PATH/questions.csv" \
  --proms "/PRIVATE/PATH/proms_df.csv" \
  --output "/PRIVATE/PATH/reproduced_outputs"
```

CSV, Parquet, and Feather DataFrames are supported. Parquet and Feather require a
pandas-compatible engine such as `pyarrow`. Do not place private inputs inside the
GitHub repository.

For non-default column names, copy `reproduction_config.example.json`, edit only the
paths and column mappings, and run:

```bash
conversation-measurement-reproduce --config /PRIVATE/PATH/reproduction_config.json
```

## How the column adapter works

The direct command assumes the column names in the first table. The config version maps
five semantic segment fields, three question fields, and five PROM fields to whatever
names your files use. Extra columns are ignored: the runner reads only the mapped columns,
validates their values, renames an in-memory copy for the original plotting functions,
and never edits the input files. Segment percentages are always recalculated from the
mapped encounter and duration columns, so a stale precomputed percentage column cannot
change the figures.

If the A–D panel order must match an existing figure, add the private order to that
private config (never to the public example):

```json
{"labels": {"clinician_order": ["GROUP_1", "GROUP_2", "GROUP_3", "GROUP_4"]}}
```

## Outputs

```text
reproduced_outputs/
├── figures/
│   ├── clinicians.pdf
│   ├── overall_phase_alloc.pdf
│   ├── question_panel.pdf
│   ├── sequence.pdf
│   └── survey.pdf
├── tables/
│   ├── prom_summary.csv
│   └── prom_summary.tex
└── verification/
    ├── prom_summary_vs_paper.csv
    └── reproduction_summary.json
```

The command exits with an error when the recomputed main PROM table does not match the
value printed in the paper. The comparison file identifies every mismatched cell.

## Exact analysis details

- Internal phase labels are mapped to the eight display labels used in the paper.
- Phase percentages are normalized within encounter before averaging across encounters.
- Clinician values are displayed only as A–D. By default, their panel order follows
  first appearance in the segments file. To preserve a specific private order, set
  `labels.clinician_order` in a private config that remains outside the repository.
- Sequence profiles use K-means with `K=4`, `random_state=0`, and `n_init="auto"`.
- `VHI`, `CSI`, and `EAT` are displayed as Voice, Cough, and Swallowing.
- PROM missingness uses every PROM item row.
- QWK, MAE, and signed error exclude pairs where either score is missing.
- Half-point patient scores are rounded upward for QWK only.
- MAE and signed error use the original, unrounded patient score.
- Signed error is `patient-reported score - transcript-derived score`.
