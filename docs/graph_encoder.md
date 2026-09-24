# Pose graph encoder

`ss-check-graph-encoder` validates the first trainable model boundary after graph
preprocessing. It reads the fixed `[64, 75, 7]` caches and never accesses source video or
reruns RTMPose.

## Architecture v1

The encoder uses a 7-to-128 input projection, learned joint and temporal embeddings, and four
alternating mask-aware blocks:

1. degree-normalized spatial graph aggregation over the pinned 75-node adjacency matrix;
2. residual feed-forward transformation;
3. depthwise temporal convolution followed by pointwise channel mixing.

Masked joints are zeroed after every residual block. Joint pooling and temporal pooling are
masked means. The resulting 256-dimensional clip embedding feeds a normalized linear head
whose class count comes from the model configuration. This architecture has a stable
fingerprint derived from all shape-changing settings.

## Smoke-check acceptance

The model configuration passed with `--config` must pin the manifest SHA-256 and the graph
preprocessing fingerprint. The check then verifies:

- graph-cache preprocessing fingerprints;
- logits shape `[batch, num_classes]` and finite values;
- invariance to arbitrary values placed at masked joints;
- finite cross-entropy loss and gradients through a backward pass;
- several optimizer steps on a deterministic train-only sample;
- atomic checkpoint publication plus a full SHA-256 digest.

The smoke checkpoint proves that the data/model contract and optimization path work. Its
accuracy is not a research result. It must not be selected using the test split.

## What remains

Full training requires epoch-level train and validation loops, validation-based checkpoint
selection, early stopping, metric history and final held-out test evaluation. Split
assignments must remain unchanged.
