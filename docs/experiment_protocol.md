# Experiment protocol

## ASL Citizen Version 1.0

Use the official train.csv, val.csv and test.csv sample assignments. Do not
repartition by video, create new ratio splits, or move signers between sets.
All preprocessing parameters learned from data must be fit on training only;
choose hyperparameters and checkpoints on validation, then report test results.

The 2,731-class vocabulary is derived from exact training Gloss strings and
shared across all splits. ASL-LEX Code is an optional annotation and cannot
replace the gloss class identifier. Preserve source CSV hashes, label map,
split artifact, generated configuration and validation reports with every run.

Start with a small, explicitly named development subset for checking tensor
shapes and pose extraction. Choose its classes using training data only and
retain original signer/split membership. Such a subset is not the full official
benchmark; do not compare its metrics directly with full-dataset paper results.

For the later recognition/dictionary-retrieval milestone, report at least
top-1, top-5 and top-10 accuracy (recall@K for one correct gloss per query).
MRR may supplement these for ranking analysis. Training/evaluation modules
are currently scaffolding, not an implemented benchmark reproduction.

ASL and Vietnamese Sign Language are different languages. Experiments using
ASL Citizen establish results on ASL only. Any ASL-to-VSL transfer experiment
requires VSL training/validation data and an independent held-out VSL test.

## VSL400

Existing VSL400 signer-disjoint search remains available through its separate
configuration. Its 80/10/10 allocation is a project-defined protocol unless an
official allocation is supplied; do not claim it reproduces an author split.

References:

- https://papers.nips.cc/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf
- https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/
