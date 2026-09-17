# ASL Citizen top-200 graph encoder

Notebook 06 validates the first trainable model boundary after graph preprocessing. It reads
the fixed `[64, 75, 7]` caches and never accesses source video or reruns RTMPose.

## Architecture v1

The encoder uses a 7-to-128 input projection, learned joint and temporal embeddings, and four
alternating mask-aware blocks:

1. degree-normalized spatial graph aggregation over the pinned 75-node adjacency matrix;
2. residual feed-forward transformation;
3. depthwise temporal convolution followed by pointwise channel mixing.

Masked joints are zeroed after every residual block. Joint pooling and temporal pooling are
masked means. The resulting 256-dimensional clip embedding feeds a 200-class normalized
linear head. This architecture has a stable fingerprint derived from all shape-changing
settings.

## Notebook 06 acceptance checks

`06_asl_citizen_top200_graph_encoder_check.ipynb` refuses to start until notebook 05 reports
6,146 completed caches and zero failures. It then checks:

- graph-cache preprocessing fingerprints;
- logits shape `[batch, 200]` and finite values;
- invariance to arbitrary values placed at masked joints;
- finite cross-entropy loss and gradients through a backward pass;
- several optimizer steps on a deterministic train-only sample;
- atomic checkpoint publication plus a full SHA-256 digest.

The smoke checkpoint proves that the data/model contract and optimization path work. Its
accuracy is not a research result. It must not be selected using the test split.

## What remains after notebook 06

Full training requires epoch-level train and validation loops, validation-based checkpoint
selection, early stopping, metric history and final held-out test evaluation. The official
train/validation/test assignments must remain unchanged.
