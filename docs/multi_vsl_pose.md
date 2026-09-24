# Multi-VSL RTMPose extraction

This stage prepares pose data for Multi-VSL M-VSL200 in all three camera views (center, left,
right). It runs in [notebook 10](../notebooks/10_multi_vsl_rtmpose_pose_extraction.ipynb) and
reuses the dataset-neutral `ss-extract-pose` and `ss-prepare-pose-graph` commands.

## Source data

- Metadata from the authors' [repository](https://github.com/Etdihatthoc/Multi-VSL_WACV_2025),
  `data/label_1_200/`:
  - `{train,val,test}_1_200_center_ord1.csv`: one center clip per row (`name`, 0-based
    `label`, `video_lb_id`);
  - `{train,val,test}_1_200_three_view_ord1.csv`: one recording per row, with the
    synchronized `center`, `left` and `right` filenames and `label`.

  The signer comes from `_signerNN_` in each filename. `ord1` is the primary take.
- Videos: the Google Drive folder linked in the authors' README holds only a **1,000-video
  sample** spread over all 1,000 glosses, 30 signers and three views (checked 2026-09-25 with
  both gdown and the Drive API). It contains 52 of the 4,311 videos the top-50 three-view
  selection needs and no complete triplet, so the full release (about 84,000 videos) must be
  requested from the authors. Sampled center clips are portrait MPEG-4, 566–606 × 720, about
  30 fps, 2–4 s long. Faces are pixelated by the authors.
- The public metadata has numeric labels only; no Vietnamese gloss text is published.
  `VSL_NNN` is a display label for source label `NNN-1`.

## Class selection

Classes present in all three official splits are ranked by **official center training clip
count only**, ties broken by ascending source label. The ranking never depends on the view
mode, so the pose data and the RGB baseline (notebook 09) use the same class list.
`--classes 0` keeps every class. The new `class_index` is the rank (0 = first); `gloss_id`
keeps the source label.

148 of the 199 classes have exactly 22 center training clips, so the default top-50 is in
practice the 50 lowest source labels among those classes (labels 2–72). Treat it as a
deterministic development subset, not as a frequency-based vocabulary.

## Views

`--views three_view` (default) reads the official triplets: every recording contributes a
center, left and right clip that share one `instance_id`, signer, label and split. Every
triplet's center clip is checked against the center list for the same split, label and
signer. `--views center` keeps only the center list.

Some center clips have no published triplet (59 training clips among the top 50), so they
are absent from the three-view data but still used by the center-only RGB baseline.

## Split

The official split is used unchanged and verified to be signer-disjoint. Nothing is
re-split or re-balanced; all views of a recording stay in the same split.

| Split | Signers | Signer IDs | Recordings (top 50) | Videos, 3 views (top 50) |
| --- | ---: | --- | ---: | ---: |
| Train | 20 | 01 02 05 06 07 09 13 14 15 17 18 19 20 22 23 24 26 28 29 30 | 1,041 | 3,123 |
| Validation | 4 | 03 08 27 31 | 199 | 597 |
| Test | 4 | 04 10 16 21 | 197 | 591 |

All 199 classes in three views give 16,989 videos. Use validation for checkpoint selection
and early stopping; evaluate test once, after the configuration is frozen.

## Top-50 classes

Recordings per split (each recording has three videos). The same table is written to
`prepared/selection.json` and printed by notebook 10.

| Rank | class_index | Source label | Display | Train | Val | Test |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 0 | 2 | VSL_003 | 21 | 4 | 4 |
| 2 | 1 | 3 | VSL_004 | 21 | 4 | 4 |
| 3 | 2 | 4 | VSL_005 | 21 | 3 | 2 |
| 4 | 3 | 6 | VSL_007 | 20 | 4 | 4 |
| 5 | 4 | 7 | VSL_008 | 21 | 4 | 4 |
| 6 | 5 | 8 | VSL_009 | 21 | 4 | 4 |
| 7 | 6 | 9 | VSL_010 | 21 | 4 | 4 |
| 8 | 7 | 10 | VSL_011 | 21 | 4 | 4 |
| 9 | 8 | 11 | VSL_012 | 21 | 4 | 4 |
| 10 | 9 | 12 | VSL_013 | 21 | 4 | 4 |
| 11 | 10 | 16 | VSL_017 | 21 | 4 | 4 |
| 12 | 11 | 18 | VSL_019 | 21 | 4 | 4 |
| 13 | 12 | 19 | VSL_020 | 21 | 4 | 3 |
| 14 | 13 | 20 | VSL_021 | 21 | 4 | 4 |
| 15 | 14 | 21 | VSL_022 | 21 | 4 | 4 |
| 16 | 15 | 22 | VSL_023 | 21 | 4 | 4 |
| 17 | 16 | 23 | VSL_024 | 21 | 4 | 4 |
| 18 | 17 | 26 | VSL_027 | 21 | 4 | 4 |
| 19 | 18 | 27 | VSL_028 | 21 | 4 | 4 |
| 20 | 19 | 28 | VSL_029 | 21 | 4 | 4 |
| 21 | 20 | 29 | VSL_030 | 21 | 4 | 4 |
| 22 | 21 | 30 | VSL_031 | 21 | 4 | 4 |
| 23 | 22 | 31 | VSL_032 | 21 | 4 | 4 |
| 24 | 23 | 32 | VSL_033 | 21 | 4 | 4 |
| 25 | 24 | 34 | VSL_035 | 21 | 4 | 4 |
| 26 | 25 | 35 | VSL_036 | 21 | 4 | 4 |
| 27 | 26 | 36 | VSL_037 | 21 | 4 | 4 |
| 28 | 27 | 37 | VSL_038 | 21 | 4 | 4 |
| 29 | 28 | 38 | VSL_039 | 21 | 4 | 4 |
| 30 | 29 | 39 | VSL_040 | 21 | 4 | 4 |
| 31 | 30 | 40 | VSL_041 | 21 | 4 | 4 |
| 32 | 31 | 41 | VSL_042 | 21 | 4 | 4 |
| 33 | 32 | 43 | VSL_044 | 21 | 4 | 4 |
| 34 | 33 | 45 | VSL_046 | 21 | 4 | 4 |
| 35 | 34 | 49 | VSL_050 | 21 | 4 | 4 |
| 36 | 35 | 51 | VSL_052 | 21 | 4 | 4 |
| 37 | 36 | 53 | VSL_054 | 21 | 4 | 4 |
| 38 | 37 | 55 | VSL_056 | 21 | 4 | 4 |
| 39 | 38 | 57 | VSL_058 | 21 | 4 | 4 |
| 40 | 39 | 59 | VSL_060 | 21 | 4 | 4 |
| 41 | 40 | 60 | VSL_061 | 20 | 4 | 4 |
| 42 | 41 | 61 | VSL_062 | 20 | 4 | 4 |
| 43 | 42 | 62 | VSL_063 | 20 | 4 | 4 |
| 44 | 43 | 63 | VSL_064 | 20 | 4 | 4 |
| 45 | 44 | 64 | VSL_065 | 20 | 4 | 4 |
| 46 | 45 | 66 | VSL_067 | 20 | 4 | 4 |
| 47 | 46 | 67 | VSL_068 | 20 | 4 | 4 |
| 48 | 47 | 70 | VSL_071 | 20 | 4 | 4 |
| 49 | 48 | 71 | VSL_072 | 21 | 4 | 4 |
| 50 | 49 | 72 | VSL_073 | 21 | 4 | 4 |

## Manifest

`ss-prepare-multi-vsl-pose list-videos` writes the filenames the selection needs, so only
those clips are fetched. `ss-prepare-multi-vsl-pose build` then writes, under `prepared/`:

- `manifest.csv` / `manifest.parquet`: the shared manifest contract, one row per video, with
  `view`, `instance_id = multi-vsl-<center video_lb_id>` and
  `sample_id = multi-vsl-<center video_lb_id>-<stem>`. Center sample IDs are identical to the
  RGB baseline, so both branches can be joined per clip;
- `labels.json`, `split.json` (strategy `official`, signer IDs, recording and video
  counts), `selection.json` (rule, class table and SHA-256 of each source CSV);
- `validation_report.json`: duplicate samples, complete view groups, consistent labels and
  signers per recording, split membership, signer overlap, missing or empty files and, with
  `--level probe`, ffprobe frame count, fps and size.

The command exits non-zero on any validation error.

## Pose extraction

`ss-extract-pose extract` processes every decoded frame of every video, independently per
view:

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

`configs/preprocessing/coco_wholebody_75_t64.yaml` turns each raw sequence into `[64, 75, 7]`
features: layout `coco_wholebody_75_v1` (13 body, 2 × 21 hand, 20 face joints), confidence
threshold 0.30, interpolation of gaps up to 3 frames, shoulder/hip normalization, uniform
sampling to 64 frames, and channels x, y, confidence, velocity and bone vectors. Each view is
a separate graph sample; group them by `instance_id` for multi-view models. Notebook 10 pins
the manifest SHA-256, extractor fingerprint and split counts before running it.

## Known limitations

- Face keypoints come from pixelated faces and should be considered unreliable. Raw caches
  keep all 133 points, so a body-and-hands layout can be derived later without rerunning
  RTMPose.
- Only two center clips were inspected; left/right resolution and framing are validated per
  clip at run time.
- The Drive API download path and the RTMPose run itself are verified only on Colab; local
  tests cover selection, manifest, validation, cache and graph stages with a fake
  extractor.
