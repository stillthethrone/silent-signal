# Shared data contract: ASL Citizen and VSL400

## Unit of data

One physical video is a **sample**. In VSL400, synchronized front, left, and
right samples sharing the same six-digit source video ID form one **instance**.
In ASL Citizen, every video is its own instance, with `view=single` and
`sample_id=instance_id=asl_citizen:<original filename>`.
ASL Citizen imports official split membership; VSL400 allocates entire signers.

## Manifest schema

| Field | Type | Meaning |
| --- | --- | --- |
| sample_id | string | Unique physical video key; dataset-specific construction above. |
| instance_id | string | Synchronized VSL400 instance, or the ASL sample key. |
| video_id | string | VSL400 padded ID; ASL Citizen exact filename including extension. |
| signer_id | string | VSL400 padded ID; ASL Citizen exact Participant ID string. |
| gloss_id | string | Source vocabulary ID; never assumed contiguous. |
| gloss_name | string | Canonical UTF-8 gloss. |
| class_index | integer | Stable contiguous zero-based training label. |
| view | enum | VSL400: front, left, right. ASL Citizen: single. |
| video_path | string | POSIX-style path relative to the configured dataset root. |
| metadata fields | nullable scalar | Values declared by release metadata. |
| measured media fields | nullable scalar | Values measured by ffprobe. |
| is_valid | boolean | False when any record-level error exists. |
| validation_errors | list of strings | Stable machine-readable error codes. |
| split | nullable enum | train, validation, or test. |
| asl_lex_code | nullable string | ASL source annotation; not necessarily unique per gloss. |

The optional `asl_lex_code` column is an additive schema change. Readers accept
older manifests without it and set it to null. Existing VSL400 identifiers and
class indexing remain unchanged. ASL labels are the lexicographically sorted
exact `Gloss` strings in train.csv, without case folding; validation and test
must use this vocabulary. ASL `gloss_id` equals the source `Gloss`, not ASL-LEX Code.

## Required invariants

1. sample_id is unique.
2. Each VSL400 instance has exactly one front, left and right sample; each ASL instance has one single-view sample.
3. Samples in an instance have identical signer and gloss labels.
4. A signer occurs in exactly one split.
5. A complete instance occurs in exactly one split.
6. Every video path remains within the configured dataset root.
7. Source gloss_id and class_index are kept separate.

## Validation levels

- metadata: validates contracts, expected counts, paths, existence, and file
  size. This is the safe first run.
- probe: additionally reads stream headers with ffprobe and compares dimensions,
  fps, duration, and frame counts.
- decode: additionally decodes every frame with ffmpeg; use only after a
  successful probe pass.

Global count mismatches appear in the report but cannot be attached to a single
manifest row. Record-level errors set is_valid to false.

## Split protocol

ASL Citizen requires `split.strategy=official`. All three CSVs are required and
their per-video assignments are preserved even at build-manifest time. No JSON
allocation override, ratios or random fallback are allowed. `create-splits`
rereads the source CSVs and rejects altered, missing or added manifest rows.
Measured video properties may change; identities, labels and source memberships
may not. The split artifact records SHA-256 digests of each source CSV and the
normalized assignments, exact counts and observed ratios. Seed and optimization
score are null because no search is performed.

The canonical ASL release has 40,154/10,304/32,941 clips and 35/6/11 signers in
train/validation/test. Configured global and per-split counts are verified.
ASL recording FPS and dimensions are measured, not constrained to VSL400 values.

### VSL400

The default is deterministic 80/10/10 signer-disjoint splitting with seed 42.
For 28 signers, largest-remainder allocation yields 22 train, 3 validation, and
3 test signers. Multiple seeded candidates are scored using:

- deviation from target instance ratios;
- missing glosses per split;
- deviation from the global gloss distribution.

If the dataset release provides an authoritative signer allocation, place it at
data/splits/vsl400_official.json; it is validated for overlap, missing, and
unknown signers before use.
