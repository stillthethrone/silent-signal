# Silent Signal

Reproducible isolated Vietnamese Sign Language (VSL) recognition research with
VSL400 and Multi-VSL. The implemented milestones prepare and validate VSL400
with signer-disjoint splits, train a frozen VideoMAE V2 + RGB Transformer
baseline on the official Multi-VSL M-VSL200 split, and provide reproducible
offline whole-body pose extraction with explicit RTMDet and RTMPose-L 384x288
artifacts. A pose graph preprocessing stage and a tested
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

## Multi-VSL Vietnamese baseline

The standalone
[Multi-VSL 50-class RGB baseline notebook](notebooks/09_multi_vsl_top50_videomaev2_rgb_transformer_baseline.ipynb)
uses the official M-VSL200 center-view metadata and keeps its signer-disjoint
train/validation/test assignment. It ranks eligible classes using training-set
clip counts only, verifies every selected official video, freezes VideoMAE V2,
and trains a compact 64-dimensional RGB Transformer with early stopping.
Source videos remain in the temporary Colab runtime; only reproducibility
manifests, logs, checkpoints, predictions, reports, and figures are saved to
Google Drive.

Open it in
[Google Colab](https://colab.research.google.com/github/stillthethrone/silent-signal/blob/feat/multi-vsl-baseline/notebooks/09_multi_vsl_top50_videomaev2_rgb_transformer_baseline.ipynb).
The Drive folder linked by the authors holds only a 1,000-video sample (20 of
the 1,496 clips this baseline needs), so the full release must be requested
from the authors; the notebook validates required files before training rather
than silently using a partial download. Public metadata exposes numeric labels but not Vietnamese
gloss text; reports therefore use stable `VSL_NNN` display labels.

## Multi-VSL RTMPose extraction

[Notebook 10](notebooks/10_multi_vsl_rtmpose_pose_extraction.ipynb) extracts
RTMPose-L WholeBody keypoints for the same 50 M-VSL200 classes as the RGB
baseline (or all 199), in all three synchronized views (center, left, right).
It keeps the official signer-disjoint split (20/4/4 signers; 1,041/199/197
recordings, 4,311 videos for the top 50), copies only the required videos
from the full release (the public Drive sample is not enough) into the
temporary runtime, and writes the manifest,
split, class list, raw pose caches and `[64, 75, 7]` graph tensors to Google
Drive. Locally, the same selection is available as:

```powershell
uv run ss-prepare-multi-vsl-pose list-videos --metadata-root <repo>/data/label_1_200 --output required.txt
uv run ss-prepare-multi-vsl-pose build --metadata-root <repo>/data/label_1_200 `
  --video-root D:/datasets/Multi-VSL/videos --output-root data/multi_vsl_pose --level probe
```

See [the Multi-VSL pose contract](docs/multi_vsl_pose.md) for the selection rule,
the 50-class list, split table, extraction steps and known limitations.

## VSL400 RTMPose extraction

[Notebook 11](notebooks/11_vsl400_rtmpose_pose_extraction.ipynb) extracts
RTMPose-L WholeBody keypoints for 50 VSL400 glosses in all three views. It reads
the extracted release from Google Drive, builds the full 400-gloss manifest and
the signer-disjoint 22/3/3 split first, then selects glosses by training
recordings only (or an explicit `GLOSS_IDS` list) without re-splitting. Only the
selected videos are copied into the runtime, re-validated with ffprobe and
passed to `ss-extract-pose` and `ss-prepare-pose-graph`. The subset step is also
available locally:

```powershell
uv run ss-select-classes --manifest data/manifests/vsl400.parquet --output-root data/subsets/vsl400_top50 --classes 50
```

See [the VSL400 pose contract](docs/vsl400_pose.md) for details.

## Kaggle VSL all-class RGB + MediaPipe fusion

[Notebook 13](notebooks/13_kaggle_vsl_allclass_rgb_preparation.ipynb) builds a
new manifest for every canonical class that has enough official-train samples
to make train/validation and at least one official-test clip. It reuses the
complete MediaPipe archive produced by notebook 11, downloads only matching
cropped front-view RGB clips with resumable coalesced ZIP ranges, and freezes
the pinned VideoMAE V2 Base backbone into compact `[8, 768]` float16 tokens.
The actual class count is derived from the data; neither 70 nor 472 is assumed.

[Notebook 14](notebooks/14_kaggle_vsl_allclass_rgb_pose_fusion_training.ipynb)
trains a small pose Graph-Spatial-Temporal encoder and RGB temporal adapter with
gated late fusion. It logs batch/epoch progress, uses class-balanced loss and
modality dropout, stops on validation loss, restores the best checkpoint, and
only then evaluates the official test split. Run notebook 13 once before
notebook 14; subsequent runs reuse the Drive packs and resumable checkpoints.
Because this public processed release has no signer IDs, its validation split
is deterministic sample-disjoint, not signer-disjoint.

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
