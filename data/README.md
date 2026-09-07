# Data policy

ASL Citizen source videos and metadata should remain outside Git. Point
ASL_CITIZEN_ROOT to the extracted ASL_Citizen directory, or pass --root with
configs/dataset/asl_citizen.yaml. The official train/val/test CSVs are source
data: do not edit or reshuffle them. Generated ASL manifests, labels and split
artifacts are ignored by Git and have dataset-specific filenames.

ASL Citizen is released for non-commercial research with restrictions on
redistribution. See the official dataset license:
https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/.
Keep downloaded ZIPs, videos and derived caches in private research storage.

VSL400 videos are controlled-access research data. Keep the source release
outside this repository and point VSL400_ROOT to it. Do not copy videos,
participant data, extracted frames, or pose caches into Git.

Tracked or reproducible metadata belongs in:

- manifests/: one row per physical video clip.
- labels/: stable source-gloss to contiguous class-index mappings.
- splits/: signer allocations and protocol definitions.

Local-only data belongs in:

- raw/: immutable source material, if a local copy is necessary.
- processed/: generated frames, pose caches, or intermediate features.

The preparation command writes generated files to these directories; paths can
be changed in configs/dataset/asl_citizen.yaml or configs/dataset/vsl400.yaml.
