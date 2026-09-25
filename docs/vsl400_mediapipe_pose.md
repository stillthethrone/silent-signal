# VSL400 pose branch on MediaPipe keypoints

[Notebook 12](../notebooks/12_vsl400_mediapipe_pose_transformer.ipynb) trains the pose branch
of the RGB–pose design (Graph Encoder → Spatial Transformer → Joint Pooling → Temporal
Transformer) on 70 VSL400 glosses, front view, using MediaPipe keypoints that the Kaggle
uploader already extracted. It is the fast path to a working end-to-end model; the RTMPose
three-view pipeline (notebook 11) remains the controlled alternative.

## Data

- Source: Kaggle `nguyenanfms/vsl-vietnamese-sign-language-v2`, version 8, a third-party
  redistribution of VSL400. Confirm permission with the VSL400 maintainers before use; the
  notebook requires `CONFIRM_KAGGLE_VSL400_PERMISSION = True`.
- Keypoints: `processed/processed/keypoints_splited/<split>/<gloss>/<video_id>.npy`,
  `[T, 76, 3]`, produced by [nguyenanfms/VSL-VietnameseSignLanguage](https://github.com/nguyenanfms/VSL-VietnameseSignLanguage):
  videos trimmed to the active signing segment (elbow-angle TBL) and cropped to 224 × 224,
  then MediaPipe Holistic (model complexity 1).
  - joints 0–32: MediaPipe Pose in its own order; 33: neck (mean of the shoulders);
  - joints 34–75: the 21 hand landmarks with left (`_0`) and right (`_1`) **interleaved**, in
    the order wrist, index tip→MCP, middle, ring, little, thumb tip→CMC;
  - the body is normalized to [-0.5, 0.5] of a 1.6× box around nose, shoulders, hips and
    neck; **each hand is normalized to its own box**; missing detections are (0, 0, 0).
- Only the *canonical* keypoints are used. The uploader's `processed_augmented` copies are
  imputed and perturbed, so their missing-joint masks cannot be recovered.
- The uploader's own train/test folders are ignored; they have no validation split.

## Split and gloss selection

Signer IDs are not in the keypoint files, so the notebook reads the VSL400 metadata JSONs of
the seven raw parts from the same archive, builds the full 400-gloss manifest and creates the
project's signer-disjoint 80/10/10 split (22 / 3 / 3 signers, seed 42; shared with notebook
11). `ss-select-classes --classes 70` then keeps the 70 glosses with the most training
recordings that occur in every split, without re-splitting.

`ss-fetch-vsl400-kaggle keypoints` matches each front-view clip to a keypoint file by video
ID **and** gloss folder (Unicode-normalized), skipping the uploader's internet-sourced clips
whose IDs can collide. Unmatched clips are listed in the packed file's metadata, and the run
fails below `MIN_KEYPOINT_COVERAGE` (default 95 %).

## Features

Layout `mediapipe_upper68_v1` (68 nodes): 25 upper-body pose landmarks (nose to hips), the
neck and both hands. Knees, ankles and feet are dropped: they lie outside the 224 × 224 crop
and were stored without visibility scores. Edges follow MediaPipe's pose and hand
connections plus neck links and a body-wrist ↔ hand-wrist bridge.

Per clip: joints are observed when not (0, 0, 0); gaps of up to 3 frames are interpolated;
the sequence is resampled to 64 frames; channels are x, y, z, velocity (3) and bone vectors
(3) = `[64, 68, 9]`. Bones stay inside the body or inside one hand, because the hands use
their own coordinate frames. Training adds a random temporal window (80–100 %), a ±10°
rotation, 0.9–1.1 scaling, a small shift and noise; evaluation is deterministic.

## Model

`PoseGraphTransformer` (≈2.2 M parameters at the defaults):

| Stage | Setting |
| --- | --- |
| Input projection + joint embedding | 9 → 128 |
| Graph Encoder | 2 mask-aware residual graph blocks over the normalized adjacency |
| Spatial Transformer | 2 layers × 4 heads over the 68 joints of each frame |
| Joint Pooling | masked attention pooling to one token per frame |
| Temporal Transformer | 3 layers × 4 heads, width 256, CLS token, learned time embedding |
| Head | LayerNorm + linear, 70 classes; the CLS output is the pose embedding `[B, 256]` |

Masked joints and frames cannot change the output (tested). `encode()` also returns the
per-frame tokens for the planned cross-attention with the RGB branch.

## Training and evaluation

AdamW (weight decay 0.05, none on norms/embeddings), linear warmup (5 epochs) then cosine,
label smoothing 0.1, dropout 0.2, gradient clipping, fp16 autocast on GPU. Validation loss
drives checkpoint selection and early stopping (patience 15). `last_checkpoint.pt` is
written every epoch and a rerun resumes only when data and settings match.

The best checkpoint is evaluated on validation, and on test only with `--run-test`
(`RUN_TEST` in the notebook). Outputs: `report.json`, `history.json`,
`predictions_<split>.csv` (with top-5), `per_class_<split>.csv`, `confusion_<split>.csv` and
figures.

## Training curves

`history.json` records, per epoch, train and validation loss, top-1, top-5 and macro-F1,
the learning rate and the epoch time. Train values are running metrics over augmented
batches with dropout active, so they understate accuracy on clean training clips.
`ss-plot-training --run-root <run>` (or notebook step 6a, which also works mid-training)
draws a 2 × 3 figure — loss, top-1 with chance level, top-5, macro-F1, generalization gap
and learning rate, with the best epoch marked — and writes `training_summary.json` with
heuristic findings:

| Finding | Rule |
| --- | --- |
| `overfitting` | ≥ 3 epochs after the best, validation loss up > 5 % while train loss down > 5 % |
| `large_generalization_gap` | train − validation top-1 > 0.25 at the best epoch |
| `underfitting` | train top-1 < 0.5 at the best epoch with a gap < 0.1 |
| `still_improving` | all requested epochs used and the best epoch is one of the last two |
| `noisy_validation` | validation top-1 moves > 5 points per epoch beyond its trend (last 10 epochs) |

## Verification status

Unit and integration tests cover the layout against the uploader's joint order, the features,
the model's mask invariance and gradients, the Kaggle keypoint matching (NFD gloss folders,
colliding IDs, missing clips, augmented decoys), and training with resume and early
stopping. The whole notebook was executed locally against a Kaggle-shaped stub archive. The
real Kaggle archive, the real keypoints and GPU training are verified only on Colab.
