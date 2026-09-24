# Multi-VSL RTMPose extraction

This stage prepares pose data for the Multi-VSL M-VSL200 center view. It runs in
[notebook 10](../notebooks/10_multi_vsl_rtmpose_pose_extraction.ipynb) and reuses the
dataset-neutral `ss-extract-pose` and `ss-prepare-pose-graph` commands.

## Source data

- Metadata: `data/label_1_200/{train,val,test}_1_200_center_ord1.csv` from the authors'
  [repository](https://github.com/Etdihatthoc/Multi-VSL_WACV_2025). Columns: `name`,
  `label` (0-based), `video_lb_id`. The signer comes from `_signerNN_` in the filename.
- Videos: the authors' single Google Drive folder (about 84,000 files, all views). Sampled
  center clips are portrait MPEG-4, 566–606 × 720, about 30 fps, 2–4 s long. Faces are
  pixelated by the authors.
- Public metadata has numeric labels only; `VSL_NNN` is a display label for source label
  `NNN-1`.

## Class selection

Classes present in all three official splits are ranked by **official training clip count
only**, ties broken by ascending source label. `--classes 0` keeps every class.
The new `class_index` is the rank (0 = first), matching the class order of the RGB baseline;
`gloss_id` keeps the source label.

148 of the 199 center-view classes have exactly 22 training clips, so the default top-50 is
in practice the 50 lowest source labels among those classes (labels 2–72). Treat it as a
deterministic development subset, not as a frequency-based vocabulary.

## Split

The official split is used unchanged and verified to be signer-disjoint. Nothing is
re-split or re-balanced.

| Split | Signers | Signer IDs | Clips (top-50) | Clips (all 199) |
| --- | ---: | --- | ---: | ---: |
| Train | 20 | 01 02 05 06 07 09 13 14 15 17 18 19 20 22 23 24 26 28 29 30 | 1,100 | 4,316 |
| Validation | 4 | 03 08 27 31 | 199 | 792 |
| Test | 4 | 04 10 16 21 | 197 | 791 |

Top-50 has 22 training, 3–4 validation and 2–4 test clips per class. Use validation for
checkpoint selection and early stopping; evaluate test once, after the configuration is
frozen.

## Manifest

`ss-prepare-multi-vsl-pose list-videos` writes the filenames the selection needs, so only
those clips are fetched. `ss-prepare-multi-vsl-pose build` then writes, under `prepared/`:

- `manifest.csv` / `manifest.parquet`: the shared manifest contract, one row per clip, with
  `sample_id = multi-vsl-<video_lb_id>-<stem>` (identical to the RGB baseline, so both
  branches can be joined per clip) and `view = center`;
- `labels.json`, `split.json` (strategy `official`, signer IDs and counts),
  `selection.json` (rule, class table and SHA-256 of each source CSV);
- `validation_report.json`: duplicate samples, split membership, signer overlap, missing or
  empty files and, with `--level probe`, ffprobe frame count, fps and size.

The command exits non-zero on any validation error.

## Pose extraction

`ss-extract-pose extract` processes every decoded frame:

1. RTMDet-M detects person boxes; boxes scoring below 0.30 are dropped.
2. The signer box is chosen by detector score, centrality, area and IoU with the previous
   frame's box.
3. The box is enlarged 1.2× and clipped to the frame so extended hands stay inside.
4. RTMPose-L 384×288 (COCO-WholeBody) predicts 133 keypoints with scores.
5. Frames without a person are stored as zeros with `person_detected = false`.

Each clip becomes one atomic, pickle-free NPZ with pixel coordinates, scores, boxes, frame
indices, timestamps and provenance (model SHA-256, library versions, source-video SHA-256).
A cache is reused only if its sample ID, video hash and extractor fingerprint match.
Extraction also fails a clip if the decoded frame count or size differs from the probed
manifest values. Sharding (`--num-shards`, `--shard-index`) splits the work
deterministically across runtimes.

## Graph tensors

`configs/preprocessing/multi_vsl_graph.yaml` turns each raw sequence into `[64, 75, 7]`
features: layout `coco_wholebody_75_v1` (13 body, 2 × 21 hand, 20 face joints), confidence
threshold 0.30, interpolation of gaps up to 3 frames, shoulder/hip normalization, uniform
sampling to 64 frames, and channels x, y, confidence, velocity and bone vectors. Notebook 10
pins the manifest SHA-256, extractor fingerprint and split counts before running it.

## Known limitations

- Face keypoints come from pixelated faces and should be considered unreliable. Raw caches
  keep all 133 points, so a body-and-hands layout can be derived later without rerunning
  RTMPose.
- The Drive API download path and the RTMPose run itself are verified only on Colab; local
  tests cover selection, manifest, validation, cache and graph stages with a fake
  extractor.
