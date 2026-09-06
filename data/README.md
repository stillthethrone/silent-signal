# Data policy

VSL400 videos are controlled-access research data. Keep the source release
outside this repository and point VSL400_ROOT to it. Do not copy videos,
participant data, extracted frames, or pose caches into Git.

Tracked or reproducible metadata belongs in:

- manifests/: one row per physical camera clip.
- labels/: stable source-gloss to contiguous class-index mappings.
- splits/: signer allocations and protocol definitions.

Local-only data belongs in:

- raw/: immutable source material, if a local copy is necessary.
- processed/: generated frames, pose caches, or intermediate features.

The preparation command writes generated files to these directories; paths can
be changed in configs/dataset/vsl400.yaml.
