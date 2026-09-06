# Silent Signal

Reproducible isolated sign-language recognition research with ASL Citizen
(American Sign Language) and VSL400 (Vietnamese Sign Language). The implemented
milestone prepares and validates data, imports ASL Citizen's official splits,
and supports VSL400 signer allocation. Pose extraction and model training remain
unimplemented scaffolding.

## ASL Citizen preparation

The adapter reads the official release's `splits/train.csv`, `splits/val.csv`
and `splits/test.csv`, preserving each video's filename, participant ID, gloss
and split. It uses the 2,731 training glosses as class labels. `ASL-LEX Code`
is retained as an optional annotation, not used as a unique class identifier.

ASL Citizen Version 1.0 contains 83,399 videos from 52 signers:

| Split | Videos | Signers |
| --- | ---: | ---: |
| Train | 40,154 | 35 |
| Validation | 10,304 | 6 |
| Test | 32,941 | 11 |

These assignments are imported exactly, not regenerated from percentages.
The pipeline rejects duplicate samples, signer leakage, missing split files,
unseen evaluation labels and changes to the official manifest membership.
File checks, FFmpeg validation and CSV/Parquet output are shared with VSL400.

Expected release layout:

```text
ASL_Citizen/
├── videos/
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── use.txt
```

Install once with `uv sync --extra dev`, then prepare the extracted dataset:

```powershell
uv run ss-prepare all --config configs/dataset/asl_citizen.yaml --root C:/datasets/ASL_Citizen --level metadata
uv run ss-prepare all --config configs/dataset/asl_citizen.yaml --root C:/datasets/ASL_Citizen --level probe --workers 4
```

Alternatively set `ASL_CITIZEN_ROOT` and omit `--root`. The command writes
`data/manifests/asl_citizen.csv` and `.parquet`,
`data/labels/asl_citizen_labels.json`, `data/splits/asl_citizen_official.json`,
and validation reports under `artifacts/runs/asl-citizen-preparation/`.
Always check the command exit code and report `passed`; global count errors
can occur even if every individual row has `is_valid=true`.

For Google Colab, open
[00_asl_citizen_colab_preparation.ipynb](notebooks/00_asl_citizen_colab_preparation.ipynb).
It clones the project's `dev` branch, downloads the official ZIP when enabled,
checks extraction space, imports the official CSVs, and stores preparation
outputs and resumable validation progress in Drive. Push this implementation
to the selected branch before running it. No Zenodo token is needed.

The Microsoft Download Center labels the ZIP as 42.8 GB. Archive plus extracted
files need roughly 89 GiB together, before extra working space or pose caches;
check the actual Colab disk quota before downloading. See the notebook and
[ASL Citizen implementation notes](docs/asl_citizen.md) for provenance, workflow
and resource requirements.

ASL Citizen is available under
[Microsoft's research dataset license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/)
for non-commercial, non-revenue-generating research, with restrictions on
redistribution. Download it from the
[official project page](https://www.microsoft.com/en-us/research/project/asl-citizen/).
An ASL-trained recognizer is not a Vietnamese sign-language recognizer;
transfer to VSL requires separate training and evaluation.

## Implemented milestone: VSL400 preparation

- Parse the three synchronized camera metadata files into one stable manifest.
- Support the release names front_view, left_view, right_view and the older
  public-code aliases cam_1, cam_2, cam_3.
- Preserve a three-video instance using one instance_id and canonical
  front/left/right view values.
- Validate schema, file existence, non-empty files, synchronized-view integrity,
  signer/gloss consistency, and published dataset counts.
- Optionally inspect video streams with ffprobe or fully decode with ffmpeg.
- Create a deterministic 80/10/10 signer-disjoint split. With 28 signers the
  target allocation is 22/3/3; candidate search balances instance counts and
  gloss coverage. An official split JSON takes precedence when supplied.
- Write CSV and Parquet manifests, a stable label map, split definition, and
  machine-readable validation reports.

## Requirements

- Python 3.11–3.13 (Python 3.12 is pinned for development).
- uv for the reproducible environment.
- FFmpeg on PATH only for --level probe or --level decode.
- Legitimate access to the controlled VSL400 video release.

## Setup

    uv sync --extra dev
    Copy-Item .env.example .env
    $env:VSL400_ROOT = "D:/datasets/VSL400"

Expected current release layout:

    VSL400/
    ├── front_view/
    ├── left_view/
    ├── right_view/
    ├── front_view.json
    ├── left_view.json
    ├── right_view.json
    └── gloss.csv

The legacy cam_1..3 directory/JSON names are discovered automatically.

## Run

Fast structural and file validation:

    uv run ss-prepare all --config configs/dataset/vsl400.yaml --level metadata

Inspect every video header:

    uv run ss-prepare all --config configs/dataset/vsl400.yaml --level probe --workers 8

Full decode is intentionally separate because it is expensive:

    uv run ss-prepare validate --config configs/dataset/vsl400.yaml --level decode --workers 4

Use an explicit root without setting an environment variable:

    uv run ss-prepare all --root D:/datasets/VSL400 --level metadata

Run uv run ss-prepare --help for individual build-manifest, validate,
create-splits, and summarize commands.

## Outputs

    data/
    ├── manifests/vsl400.csv
    ├── manifests/vsl400.parquet
    ├── labels/vsl400_labels.json
    └── splits/vsl400_signer_split.json

    artifacts/runs/data-preparation/
    ├── validation_report.json
    └── invalid_records.csv

data/raw, data/processed, and generated artifacts are ignored by Git. The
source release must never be committed.

## Quality checks

    uv run ruff check .
    uv run ruff format --check .
    uv run pytest

The data contract is documented in docs/data_contract.md.

## VSL400 on Google Colab

Use
[notebooks/00_vsl400_colab_preparation.ipynb](notebooks/00_vsl400_colab_preparation.ipynb)
to run preparation from Google Drive or a Git repository. The notebook persists
manifests, reports, label maps, signer splits, and resumable ffprobe results to
Drive. It expects an authorized, already extracted VSL400 directory and never
stores an access token in notebook cells. For approved Zenodo users, it can also
download the restricted multipart release directly into a Colab runtime using a
`ZENODO_TOKEN` supplied through Colab Secrets, with HTTP resume and MD5 checks.
