# Silent Signal

Reproducible isolated sign-language recognition research with ASL Citizen
(American Sign Language) and VSL400 (Vietnamese Sign Language). The implemented
milestones prepare and validate data, import ASL Citizen's official splits,
support VSL400 signer allocation, and provide reproducible offline whole-body
pose extraction with explicit RTMDet and RTMPose-L 384x288 artifacts. The
ASL Citizen top-200 path now includes graph preprocessing and a tested
Graph-Spatial-Temporal Encoder smoke-training boundary; full epoch-level
training and held-out evaluation remain future milestones.

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

## RTMPose-L WholeBody extraction

The pose pipeline uses MMPose as the framework, RTMDet for person detection and
the explicit COCO-WholeBody RTMPose-L 384x288 model for all 133 raw keypoints.
It does not use MMPose's mutable `wholebody` alias. Atomic caches retain source
coordinates and complete artifact/software provenance for later derivation of
the 75-node Graph-Spatial-Temporal Transformer layout.

Verify the local config/checkpoint files and calculate their full SHA-256 values:

```powershell
uv run ss-extract-pose verify --config configs/pose/rtmpose.yaml
```

Then run a small ASL Citizen pilot:

```powershell
uv run ss-extract-pose extract --config configs/pose/rtmpose.yaml `
  --manifest data/manifests/asl_citizen.parquet `
  --dataset-root D:/datasets/ASL_Citizen --split train --limit 100
```

See [the extraction contract and environment guide](docs/pose_extraction.md)
before downloading models or starting a production run.

For Google Colab, open
[00_asl_citizen_colab_preparation.ipynb](notebooks/00_asl_citizen_colab_preparation.ipynb).
After data validation passes, use
[03_rtmpose_wholebody_colab_check.ipynb](notebooks/03_rtmpose_wholebody_colab_check.ipynb)
to optionally stream-extract the full official ZIP directly from Microsoft,
verify the pinned OpenMMLab environment, download and hash the explicit
RTMDet-M/RTMPose-L artifacts, extract a smoke sample, inspect its raw 133-point
cache and validate resume behavior. It can also be opened directly in
[Google Colab](https://colab.research.google.com/github/stillthethrone/silent-signal/blob/dev/notebooks/03_rtmpose_wholebody_colab_check.ipynb)
after the notebook has been pushed to the `dev` branch.
It clones the project's `dev` branch, downloads the official ZIP when enabled,
checks extraction space, imports the official CSVs, and stores preparation
outputs and resumable validation progress in Drive. Push this implementation
to the selected branch before running it. No Zenodo token is needed.

To extract pose only for the 200 ASL Citizen classes with the highest ASL-LEX
2.0 subjective conversational-frequency ratings, use
[04_asl_citizen_top200_pose_extraction.ipynb](notebooks/04_asl_citizen_top200_pose_extraction.ipynb).
It is standalone: on a fresh GPU runtime it can stream-extract ASL Citizen, build
the official manifest, create the pinned OpenMMLab environment, download models,
select the subset and extract pose without running notebooks `00` or `03`. It
preserves the official splits and all clips in each selected class; it does not
rank words by ASL Citizen video count. See the
[top-200 selection contract](docs/asl_citizen_top200.md) for the exact rule,
outputs, source, license, and limitations.

After all 6,146 raw pose caches pass extraction, run
[05_asl_citizen_top200_graph_preparation.ipynb](notebooks/05_asl_citizen_top200_graph_preparation.ipynb).
It converts raw 133-keypoint sequences into resumable `[64, 75, 7]` graph tensors on Drive
without reading videos or using a GPU. The configuration pins the completed manifest and
extractor fingerprints and preserves the official splits. See the
[graph preprocessing contract](docs/graph_preprocessing.md) for the feature, mask,
normalization, and cache definitions.

After graph preparation passes, use
[06_asl_citizen_top200_graph_encoder_check.ipynb](notebooks/06_asl_citizen_top200_graph_encoder_check.ipynb)
to validate the mask-aware Graph-Spatial-Temporal Encoder, its 200-class head, backward pass,
optimizer step and atomic smoke checkpoint. This check is not full multi-epoch training; see
the [graph encoder contract](docs/graph_encoder.md).

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
