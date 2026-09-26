# Silent Signal

Reproducible isolated Vietnamese Sign Language (VSL) recognition research on
VSL400. The implemented milestones prepare and validate VSL400 with
signer-disjoint splits, select gloss subsets without re-splitting, and provide
reproducible offline whole-body pose extraction with explicit RTMDet and
RTMPose-L 384x288 artifacts, in all three camera views. A pose graph preprocessing stage and a tested
Graph-Spatial-Temporal Encoder smoke-training boundary are available; full
epoch-level pose training and held-out evaluation remain future milestones.

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

Then run a small pilot on a prepared manifest:

```powershell
uv run ss-extract-pose extract --config configs/pose/rtmpose.yaml `
  --manifest data/manifests/vsl400.parquet `
  --dataset-root D:/datasets/VSL400 `
  --output-root data/processed/pose/vsl400/rtmpose_l_coco_wholebody_384x288/raw `
  --split train --limit 100
```

See [the extraction contract and environment guide](docs/pose_extraction.md)
before downloading models or starting a production run.

Raw pose caches can then be converted into `[64, 75, 7]` graph tensors with
`ss-prepare-pose-graph` and checked with `ss-check-graph-encoder`. Both take an
explicit `--config`; see the [graph preprocessing](docs/graph_preprocessing.md)
and [graph encoder](docs/graph_encoder.md) contracts.

## VSL400 RTMPose extraction

[Notebook 11](notebooks/11_vsl400_rtmpose_pose_extraction.ipynb) extracts
RTMPose-L WholeBody keypoints for 50 VSL400 glosses in all three views. It reads
either the Kaggle redistribution (range-reading only the metadata and the
selected videos out of its ~75 GB ZIP, after you confirm permission to use it)
or an extracted release on Google Drive, builds the full 400-gloss manifest and
the signer-disjoint 22/3/3 split first, then selects glosses by training
recordings only (or an explicit `GLOSS_IDS` list) without re-splitting. Only the
selected videos are copied into the runtime, re-validated with ffprobe and
passed to `ss-extract-pose` and `ss-prepare-pose-graph`. The subset step is also
available locally:

```powershell
uv run ss-select-classes --manifest data/manifests/vsl400.parquet --output-root data/subsets/vsl400_top50 --classes 50
```

See [the VSL400 pose contract](docs/vsl400_pose.md) for details.

## VSL400 pose branch on MediaPipe keypoints

[Notebook 12](notebooks/12_vsl400_mediapipe_pose_transformer.ipynb) trains the
pose branch of the RGB–pose design (Graph Encoder with 2 graph blocks, Spatial
Transformer 2 × 4 heads, joint pooling, Temporal Transformer 3 × 4 heads,
256-d pose embedding) on 70 VSL400 glosses, front view. It reads the MediaPipe
Holistic keypoints already extracted in the Kaggle redistribution (after you
confirm permission), rebuilds the project's signer-disjoint split from the
VSL400 metadata, and saves the manifest, packed keypoints, checkpoints,
predictions, metrics and figures to Google Drive. The same steps are available
locally:

```powershell
uv run ss-fetch-vsl400-kaggle keypoints --manifest data/subsets/vsl400_top70/manifest.csv --output data/subsets/vsl400_top70/mediapipe76_front.npz
uv run ss-train-pose-transformer --manifest data/subsets/vsl400_top70/manifest.csv `
  --keypoints data/subsets/vsl400_top70/mediapipe76_front.npz --output-root artifacts/runs/pose_top70
```

[Notebook 14](notebooks/14_vsl400_mediapipe_pose_transformer_top200.ipynb) and
[notebook 15](notebooks/15_vsl400_mediapipe_pose_transformer_all400.ipynb) run the
same pipeline for 200 and all 400 glosses. See
[the MediaPipe pose-branch contract](docs/vsl400_mediapipe_pose.md).

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
