# ASL Citizen top-200 graph preprocessing

This stage consumes the 6,146 raw RTMPose-L WholeBody caches produced by notebook 04. It
does not read source videos or run a pose model.

## Contract

Each raw `[T, 133, 2]` sequence becomes one compressed graph cache with:

- `features`: `[64, 75, 7]` float32 in the channel order `x`, `y`, `confidence`,
  `velocity_x`, `velocity_y`, `bone_x`, `bone_y`;
- `joint_mask`: usable joints after confidence filtering and short-gap interpolation;
- `observed_mask`: joints directly observed by RTMPose before interpolation;
- `frame_mask`: sampled frames containing at least one usable joint;
- the normalized 75-node adjacency matrix, source frame indices and timestamps;
- class index, official train/validation/test split, raw extractor fingerprint and the
  graph-preprocessing fingerprint.

Coordinates are divided by source frame dimensions, centered using shoulders (hips or
valid-joint mean as fallbacks), and scaled by shoulder/hip distance or visible extent. Only
gaps of at most three frames bracketed by observations of the same joint are interpolated.
Invalid values remain zero and are always distinguished by masks.

## Reproducibility and resume

`configs/preprocessing/asl_citizen_graph.yaml` pins the expected manifest SHA-256, raw
extractor fingerprint, 6,146 clips, 200 classes, and official split counts. Existing graph
caches are skipped only after their sample ID and preprocessing fingerprint validate.
Stale or corrupt output fails closed unless `--overwrite` is explicit.

Run `notebooks/05_asl_citizen_top200_graph_preparation.ipynb` in Colab. This is CPU-only and
writes directly into the same persistent Drive subtree as the top-200 manifest and pose
caches. A stopped session can rerun the notebook with `OVERWRITE=False` and resume.

The resulting caches define the stable input boundary for implementing and training the
graph encoder; model training must continue to use the official splits unchanged.
