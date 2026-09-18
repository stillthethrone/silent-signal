# ASL Citizen top-50 graph Transformer

This experiment derives a frozen 50-class benchmark from the completed
ASL-LEX-ranked top-200 artifacts. It never re-splits clips. Every sample keeps
the official ASL Citizen `train`, `validation`, or `test` assignment, and the
selector rejects signer overlap and classes without coverage in all three
partitions.

## Artifact flow

```text
top-200 manifest + selection_report.json
                    |
                    v
       frozen top-50 manifest (labels 0..49)
                    |
raw 133-point pose -+-> 64 x 75 x 7 graph cache
                    |
                    v
 graph blocks -> spatial Transformer -> temporal Transformer -> 50-class head
```

The top-50 selector records actual clip counts and percentages after filtering.
The source top-200 manifest currently contains 2,933 train, 761 validation, and
2,452 test clips (47.72%, 12.38%, and 39.90%). Those are observed official
ratios, not targets. Top-50 ratios are allowed to differ.

## Commands

Create the frozen subset:

```bash
ss-select-asl-ranked-subset \
  --manifest /drive/asl_citizen_asllex_top200/manifest.csv \
  --selection-report /drive/asl_citizen_asllex_top200/selection_report.json \
  --output-root /drive/asl_citizen_asllex_top50 \
  --classes 50
```

Prepare graph caches from the already completed raw pose cache:

```bash
ss-prepare-pose-graph \
  --config configs/preprocessing/asl_citizen_top50_graph.yaml \
  --manifest /drive/asl_citizen_asllex_top50/manifest.csv \
  --pose-root /drive/asl_citizen_asllex_top200/pose/rtmpose_l_coco_wholebody_384x288/raw \
  --output-root /drive/asl_citizen_asllex_top50/graph/asl_citizen_coco_wholebody_v1/t64 \
  --report /drive/asl_citizen_asllex_top50/reports/graph_preparation_t64.json \
  --continue-on-error
```

Smoke-test forward, backward, and joint-mask invariance, then run full training:

```bash
ss-check-graph-encoder \
  --config configs/model/asl_citizen_top50_graph_transformer.yaml \
  --manifest /drive/asl_citizen_asllex_top50/manifest.csv \
  --graph-root /drive/asl_citizen_asllex_top50/graph/asl_citizen_coco_wholebody_v1/t64 \
  --checkpoint /drive/asl_citizen_asllex_top50/models/smoke.pt \
  --report /drive/asl_citizen_asllex_top50/reports/smoke.json \
  --device cuda --overwrite

ss-train-graph-transformer \
  --config configs/model/asl_citizen_top50_graph_transformer.yaml \
  --manifest /drive/asl_citizen_asllex_top50/manifest.csv \
  --graph-root /drive/asl_citizen_asllex_top50/graph/asl_citizen_coco_wholebody_v1/t64 \
  --output-root /drive/asl_citizen_asllex_top50/models/graph_transformer/seed_42 \
  --device cuda
```

The trainer uses inverse-frequency class weights computed only from `train`.
It selects `best.pt` using validation macro-F1, maintains `last.pt` for resume,
and evaluates the held-out test split only after restoring the best checkpoint.
The final report includes Top-1, Top-5, macro precision/recall/F1, per-class
metrics, and the complete confusion matrix.

For Colab, use
[`09_asl_citizen_top50_graph_transformer_training.ipynb`](../notebooks/09_asl_citizen_top50_graph_transformer_training.ipynb).
