# Experiment protocol

## General rules

Never move signers between train, validation and test. All preprocessing
parameters learned from data must be fit on training only; choose
hyperparameters and checkpoints on validation, then report test results once.
Preserve the manifest hash, label map, split artifact, generated configuration
and validation reports with every run.

Start with a small, explicitly named development subset for checking tensor
shapes and pose extraction. Choose its classes using training data only and
retain original signer/split membership. Such a subset is not a full benchmark;
do not compare its metrics directly with full-dataset paper results.

For the later recognition/dictionary-retrieval milestone, report at least
top-1, top-5 and top-10 accuracy (recall@K for one correct gloss per query).
MRR may supplement these for ranking analysis. Training/evaluation modules
are currently scaffolding, not an implemented benchmark reproduction.

## Multi-VSL M-VSL200

Use the official center-view train/validation/test CSVs unchanged. The
50-class baseline ranks classes by training clip count only; validation and
test labels are used solely to require coverage.

## VSL400

Existing VSL400 signer-disjoint search remains available through its separate
configuration. Its 80/10/10 allocation is a project-defined protocol unless an
official allocation is supplied; do not claim it reproduces an author split.
