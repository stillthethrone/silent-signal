# RTMPose-L WholeBody extraction

This milestone extracts raw 2D whole-body pose from ASL Citizen for the later
Graph-Spatial-Temporal Transformer. MMPose is the framework, RTMPose-L is the
pose estimator, and RTMDet supplies the person box required by the top-down
model. The implementation never uses the mutable `wholebody` alias.

## Fixed model identity

The production pose model is the official COCO-WholeBody RTMPose-L checkpoint
with a 288x384 `(width, height)` input. The official model index reports Hand AP
0.579 for this model, compared with 0.475 for RTMPose-M 256x192.

- Pose config:
  `configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py`
- Pose checkpoint URL:
  <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth>
- Detector config:
  `demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py`
- Detector checkpoint URL:
  <https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth>

Keep a source checkout of MMPose at a known revision because both explicit
configuration files inherit other files from that checkout. Do not copy only
the top-level Python configs.

## Environment

OpenMMLab binary compatibility is intentionally kept outside the lightweight
data-preparation lock. Create a dedicated Python 3.11 GPU environment, install
this repository into it with `python -m pip install -e .`, and install PyTorch
for the machine's CUDA runtime first. Then install a mutually compatible stack:

- MMPose 1.3.2
- MMDetection 3.x
- MMCV 2.x (`>=2.0.1`; install the wheel matching PyTorch/CUDA)
- MMEngine `>=0.9.0`
- OpenCV

MMPose's official installation guide should be treated as the source of truth:
<https://mmpose.readthedocs.io/en/latest/installation.html>. After the stack is
working, freeze its exact versions in the extraction environment. A native
Windows installation is supported, but WSL2, Linux, or a CUDA container usually
makes MMCV wheel selection easier to reproduce.

The commands below use the installed `ss-extract-pose` entry point. If the
OpenMMLab packages are installed directly into this repository's `.venv`, the
equivalent form is `uv run ss-extract-pose`.

Set these variables before invoking the CLI:

```powershell
$env:ASL_CITIZEN_ROOT = "D:/datasets/ASL_Citizen"
$env:MMPOSE_ROOT = "D:/src/mmpose"
$env:RTMPOSE_L_WHOLEBODY_CHECKPOINT = "D:/models/rtmpose-l-wholebody-384x288.pth"
$env:RTMDET_M_PERSON_CHECKPOINT = "D:/models/rtmdet-m-person.pth"
```

## Artifact verification

Download checkpoints only from the URLs recorded in
`configs/pose/rtmpose.yaml`. Calculate the complete hashes before extraction:

```powershell
ss-extract-pose verify `
  --config configs/pose/rtmpose.yaml `
  --write-lock artifacts/pretrained/rtmpose_l_384x288.lock.json
```

The lock records full SHA-256 values for the pose config, pose checkpoint,
detector config and detector checkpoint, along with installed MMPose,
MMDetection, MMCV, MMEngine, PyTorch, OpenCV and NumPy versions, the CUDA build,
CUDA availability and detected GPU names. Copy the two checkpoint hashes into
the YAML before any extraction run. The production extractor refuses unpinned
checkpoints and fails closed if a local artifact differs.

## Pilot extraction

Start with a stratified pilot rather than all 83,399 clips:

```powershell
ss-extract-pose extract `
  --config configs/pose/rtmpose.yaml `
  --manifest data/manifests/asl_citizen.parquet `
  --dataset-root D:/datasets/ASL_Citizen `
  --split train `
  --limit 100
```

The extractor performs these operations for every decoded frame:

1. RTMDet detects person boxes.
2. Low-score boxes are removed.
3. The primary signer is selected using detector confidence, centrality, area,
   and IoU continuity with the previous selected box.
4. The selected box is enlarged and clipped to the frame so extended hands are
   less likely to be truncated.
5. RTMPose-L predicts exactly 133 COCO-WholeBody points from that box.
6. Missing-person frames are retained as zero numeric rows with an explicit
   `person_detected=false` mask.

Run at least several manually chosen difficult clips with `--sample-id` in
addition to the broad pilot. Inspect hand confidence, missing-person frequency,
signer switching and border-touching boxes before starting the full run.

## Cache contract

Each cache is an atomic compressed NPZ containing no pickled objects:

```text
frame_indices       [T_raw]       int64
timestamps_seconds  [T_raw]       float64
keypoints_xy        [T_raw,133,2] float32, source-image pixels
keypoint_scores     [T_raw,133]   float32
bboxes_xyxy         [T_raw,4]     float32
bbox_scores         [T_raw]       float32
person_detected     [T_raw]       bool
metadata_json                     UTF-8 bytes
```

Raw coordinates are not normalized, smoothed, interpolated, thresholded, or
reduced to the 75-joint research layout. This separation allows downstream
preprocessing and graph topology to change without rerunning RTMPose.

The cache path is derived from SHA-256 of `sample_id`, preventing manifest path
traversal and distributing files across prefix directories. Resume accepts a
cache only when its sample ID, source-video SHA-256 and extractor fingerprint
all match. Stale or corrupt files require explicit `--overwrite`.

## Production extraction

Run one process per GPU. The CLI sorts sample IDs and assigns them by modulo, so
`--num-shards N --shard-index K` gives every process a deterministic, disjoint
selection. Give each process a separate report path; the cache root can be
shared because sample assignments do not overlap.

```powershell
ss-extract-pose extract `
  --config configs/pose/rtmpose.yaml `
  --manifest data/manifests/asl_citizen.parquet `
  --dataset-root D:/datasets/ASL_Citizen `
  --output-root data/processed/pose/asl_citizen/rtmpose_l_coco_wholebody_384x288/raw `
  --device cuda:0 `
  --num-shards 2 `
  --shard-index 0 `
  --report artifacts/runs/pose-extraction/rtmpose_l_shard_0.json `
  --continue-on-error `
  --progress-every 25
```

Run the second process with `--device cuda:1 --shard-index 1` and its own report.

The machine-readable report contains extracted, resumed and failed counts plus
every failure. A nonzero failure count produces exit code 1. Configuration,
manifest, missing video, invalid checkpoint, model-output and cache failures
produce actionable errors rather than silently skipping data.

## Transformer layout

`asl_citizen_coco_wholebody_v1` derives 75 graph nodes from the raw 133:

- 13 upper-body points: COCO-WholeBody indices 0-12
- 21 left-hand points: 91-111
- 21 right-hand points: 112-132
- 20 sparse eyebrow, eye and mouth points

The layout stores original indices, body-part identity, parent joints and graph
edges, including body-wrist to hand-root connections. Normalization, missing
joint interpolation, confidence masks, velocity and bone features belong to the
next preprocessing milestone, not to raw extraction.
