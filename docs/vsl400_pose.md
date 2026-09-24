# VSL400 RTMPose extraction

[Notebook 11](../notebooks/11_vsl400_rtmpose_pose_extraction.ipynb) extracts RTMPose-L
WholeBody keypoints for a 50-gloss VSL400 subset in all three views (front, left, right). It
reads the extracted release from Google Drive and reuses `ss-prepare`, `ss-select-classes`,
`ss-extract-pose` and `ss-prepare-pose-graph`.

## Source data

The extracted release contains `front_view/`, `left_view/`, `right_view/` (or the legacy
`cam_1..3`), one metadata JSON per view and `gloss.csv`. The configuration expects 74,259
clips, 400 glosses, 28 signers, three views per recording, 25 fps and 1080 × 1080. Unlike
Multi-VSL, `gloss.csv` provides the Vietnamese gloss text, so the class list carries real
words.

## Split

VSL400 has no official split. `ss-prepare all` assigns whole signers 80/10/10 over the
**full 400-gloss dataset** (22 / 3 / 3 signers for 28 signers), searching 5,000 seeded
candidates (seed 42) for balanced recording counts and gloss coverage; see the
[data contract](data_contract.md). The result is deterministic and stored in
`splits/vsl400_signer_split.json`. An authoritative author allocation, if one is published,
can be supplied instead.

The split is created before any gloss is selected and is never recomputed on the subset, so
the subset cannot influence which signers are held out. All three views of a recording share
one `instance_id` and therefore one split.

## Gloss selection

`ss-select-classes` keeps whole glosses that occur in train, validation and test, ranked by
**training recordings only**; ties keep the source class order (ascending `gloss_id`).
`--gloss-id` (repeatable) selects an explicit list instead, in the given order. The new
`class_index` is the rank; `gloss_id` and the Vietnamese `gloss_name` are preserved. Outputs,
under `subsets/<subset>_three_view/prepared/`:

- `manifest.csv|parquet`: every clip of the selected glosses, all views, split unchanged;
- `selection.json`: rule, source-manifest SHA-256, per-split recordings/clips/signers and the
  ranked gloss table;
- `labels.json` and `required_videos.txt` (relative paths to copy).

The exact 50 glosses depend on the dataset's per-signer coverage, so they are printed by the
notebook and recorded in `selection.json` rather than fixed here.

## Local copy and validation

Only the required videos are copied from Drive to `/content/VSL400_subset`, keeping relative
paths, through `.part` files with a size check. `ss-prepare validate` then re-checks the
subset against the local copy with subset-specific expected counts: complete view groups,
consistent signer and gloss per recording, non-empty files and, at `--level probe`, ffprobe
fps, size, frame count and duration against the release metadata. The probed values are
written back to the subset manifest.

## Pose extraction and graph tensors

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
A cache is reused only if its sample ID, video hash and extractor fingerprint match, and a
clip fails if its decoded frame count or size differs from the probed manifest. Sharding
(`--num-shards`, `--shard-index`) splits the work deterministically across runtimes.

`configs/preprocessing/coco_wholebody_75_t64.yaml` then turns each raw sequence into
`[64, 75, 7]` features: layout `coco_wholebody_75_v1` (13 body, 2 × 21 hand, 20 face
joints), confidence threshold 0.30, interpolation of gaps up to 3 frames, shoulder/hip
normalization, uniform sampling to 64 frames, and channels x, y, confidence, velocity and
bone vectors. Each view is a separate graph sample; group them by `instance_id` for
multi-view models.

## Verification status

Local tests run the whole chain (full manifest and split, subset, copy, validation, pose
extraction with a fake extractor, graph preparation) on a synthetic three-view VSL400
release. The Drive copy, ffprobe values of the real release and the RTMPose run itself are
verified only on Colab.
