# ASL Citizen top-200 frequency subset

This workflow selects 200 ASL Citizen classes using the subjective sign-frequency
ratings published in ASL-LEX 2.0, then passes the resulting manifest to the existing
RTMPose-L WholeBody extractor.

## What "most used" means

The ranking uses `SignFrequency(M)` from the official ASL-LEX 2.0 `signdata.csv`.
It is the mean response on a 1-7 scale to how frequently a sign appears in everyday
ASL conversation. It is not the number of videos per ASL Citizen class. Clip counts
are a property of dataset collection and are not used to decide word frequency.

The deterministic ordering is:

1. `SignFrequency(M)` descending;
2. ASL Citizen gloss name ascending, case-insensitive;
3. original ASL Citizen class index ascending.

ASL Citizen's `ASL-LEX Code` joins a class to ASL-LEX. It remains an annotation,
not a class ID: distinct ASL Citizen glosses sharing one code stay distinct. Classes
without a usable ASL-LEX score or without a training video are excluded before the
top 200 are taken.

In the official release, the selected 200 ASL Citizen labels contain 198 unique
ASL-LEX codes because `WHATFOR1`, `WHATFOR2`, and `WHATFOR3` share code
`C_02_054`. They remain separate source classes. `selection_report.json` records
both `unique_asl_lex_codes_selected` and every `shared_asl_lex_codes` group so this
distinction is explicit rather than silently merging labels.

The implementation was checked against the official ASL Citizen v1 split CSVs and
ASL-LEX 2.0 `signdata.csv`: the selected 200 labels contain 6,146 videos (2,933
train, 761 validation, 2,452 test). The highest frequency score is 6.963 and the
rank-200 cutoff is 5.786. These figures are validation anchors; the source hashes
in each generated report remain the authoritative record for a particular run.

## Outputs and invariants

`ss-select-asl-subset` writes the following under
`<output-root>/asl_citizen_asllex_top200/`:

```text
manifest.csv
manifest.parquet
labels.json
selection_report.json
```

The subset keeps every official train, validation, and test clip belonging to a
selected class. It does not resplit signers or sample a fixed number of clips.
Original class indices are recorded in `selection_report.json`; subset indices are
contiguous from 0 to 199. The report also records SHA-256 hashes of both source
files, the exact ranking, split counts, and excluded-class diagnostics.

Example:

```powershell
uv run ss-select-asl-subset `
  --manifest data/manifests/asl_citizen.csv `
  --asl-lex-csv C:/datasets/asl-lex-2/signdata.csv `
  --classes 200 `
  --output-root data/subsets
```

For Colab, use
[`04_asl_citizen_top200_pose_extraction.ipynb`](../notebooks/04_asl_citizen_top200_pose_extraction.ipynb).
Notebook `04` is standalone. On a fresh GPU runtime it can stream-extract the
official ASL Citizen release into `/content`, generate the canonical manifest,
create an isolated Python 3.11/OpenMMLab environment, download and hash the model
artifacts, select the subset, run a small pilot and continue into full extraction.
It does not require notebooks `00` or `03`. Dataset downloading requires explicit
acceptance of the Microsoft research license in the configuration cell. The pilot
and full extraction share one raw cache, so successful pilot videos are resumed
rather than repeated.

## Source, license, and limitations

- ASL-LEX 2.0 project and downloads: <https://asl-lex.org/download.html>
- Dataset paper: <https://doi.org/10.1093/deafed/enaa038>
- ASL-LEX database and visualization license: CC BY-NC 4.0. Cite the dataset paper
  and verify the current terms at <https://asl-lex.org/about.html> before reuse.
- The ASL-LEX raw sign videos are not covered by that reuse permission and are not
  needed by this workflow. Only the lexical-variable CSV is downloaded.

Subjective frequency is a measured rating from ASL signers, not a universal or
time-invariant corpus count. The top 200 are common ASL lexical classes within the
intersection of ASL-LEX and ASL Citizen. They are not Vietnamese Sign Language
words, and the selection should not be described as a VSL frequency list.
