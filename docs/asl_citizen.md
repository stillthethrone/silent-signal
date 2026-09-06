# ASL Citizen preparation

## Verified release

The ingestion contract was checked against the actual official ZIP central
directory and split CSVs using HTTP byte ranges, without downloading videos.
This is ASL Citizen Version 1.0, not MS-ASL or WLASL.

- Project: https://www.microsoft.com/en-us/research/project/asl-citizen/
- Download: https://www.microsoft.com/en-us/download/details.aspx?id=105253
- ZIP: https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip
- Paper: https://papers.nips.cc/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf
- Reference loader: https://github.com/microsoft/ASL-citizen-code/blob/main/I3D/aslcitizen_dataset.py
- License: https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/

Verified archive bytes: 45,924,134,223. Total uncompressed member bytes:
49,604,368,459. These are about 42.77 GiB and 46.20 GiB respectively; the ZIP
plus extraction needs about 88.97 GiB, plus safety headroom. Available Colab
storage varies; do not assume a free runtime can hold all intermediates.

The archive root is ASL_Citizen/, containing videos/, splits/ and use.txt.
The exact CSV columns are:

```csv
Participant ID,Video file,Gloss,ASL-LEX Code
```

| Source | Manifest split | Videos | Signers | Glosses |
| --- | --- | ---: | ---: | ---: |
| splits/train.csv | train | 40,154 | 35 | 2,731 |
| splits/val.csv | validation | 10,304 | 6 | 2,731 |
| splits/test.csv | test | 32,941 | 11 | 2,731 |

Across the release there are 83,399 videos, 52 signers and 2,731 glosses.
The CSVs have 2,723 distinct ASL-LEX Code values: use Gloss for class identity
and keep the ASL-LEX code only as an annotation. No mirror flag is included in
these CSVs. Do not infer mirroring or create cross-camera groups from filenames.

Reference SHA-256 hashes measured from the official CSVs on 2026-09-06:

| File | SHA-256 |
| --- | --- |
| train.csv | `87a105649b98ea577182ebfce49d59fe9ca834a2489384f6fe7bc03f14966329` |
| val.csv | `6ac78c9d551dfcc81d225a5279d7637c2891a4490851234be6e4674666e21366` |
| test.csv | `b1dee8f81204895ca1760f6bd582eec84cf390b9c97c4387e49b9cc6b906b21e` |

The implemented adapter was checked against every official CSV row for exact
filenames, participant IDs, glosses, ASL-LEX codes and split assignments. This
metadata verification did not download or assert the decodability of the videos.

## Implementation boundaries

The ASL adapter only interprets the source CSVs and filenames. It returns the
existing ManifestRecord contract. Shared manifest serializers, video integrity
checks, ffprobe/ffmpeg functions, reports and CLI remain in use.

VSL400's JSON parsing, six-digit filenames, padded signer identifiers and
signer-allocation search remain specific to its adapter/configuration. Existing
VSL configuration and notebook remain supported. The command-line default
configuration stays VSL400 for compatibility; pass the ASL config explicitly.

ASL has one video per instance. RGB + pose dual-stream models can still use
two representations of the same video; synchronized-camera multiview learning
is not provided by this dataset. Pose extraction and training are future work.

## Running on Colab

1. Push the ASL implementation to the Git branch selected by the notebook.
2. Open notebooks/00_asl_citizen_colab_preparation.ipynb on Colab web.
3. Mount Drive and configure repository, branch, dataset and result paths.
4. Run project installation, then explicitly enable the official ZIP download.
5. Check space and extract to /content, retaining the official splits and use.txt.
6. Run metadata preparation; review counts, labels, split provenance and report.
7. Run batched ffprobe verification. Progress is cached in the ASL Drive folder.
8. Optionally decode videos for stronger corruption detection. A limited decode
   batch is a partial check, not evidence that the whole dataset has passed.

The notebook writes its generated config and all output paths to the selected
Drive result folder before invoking the CLI. Source videos in /content are
temporary and disappear when the runtime is deleted. A saved report/cache in
Drive does not preserve the source videos. Keep different dataset runs separate.

## Reproducibility

Keep the source CSVs intact. The official split JSON includes source CSV SHA-256
hashes and a deterministic hash of sample identities, labels and assignments.
The label map is sorted from training glosses and reused for validation/test.
The CLI rejects attempts to reassign signers or silently discard source rows.

Metadata-only validation checks file presence and size but cannot establish
video decodability. A full dataset pass requires a successful complete probe
or decode run and a report with passed=true. Global count errors may coexist
with rows marked is_valid=true; always inspect the report and process exit code.

Small test fixtures in this repository are synthetic. Their counts are explicit
and must not be presented as results on the full ASL Citizen benchmark.
