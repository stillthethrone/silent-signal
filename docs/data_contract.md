# VSL400 data contract

## Unit of data

One physical video is a **sample**. The synchronized front, left, and right
samples sharing the same six-digit source video ID form one **instance**.
Splitting is performed by signer, never by sample or camera.

## Manifest schema

| Field | Type | Meaning |
| --- | --- | --- |
| sample_id | string | Unique video-ID and view key. |
| instance_id | string | Shared key for the three synchronized views. |
| video_id | string | Zero-padded source video ID. |
| signer_id | string | Zero-padded signer ID. |
| gloss_id | string | Source vocabulary ID; never assumed contiguous. |
| gloss_name | string | Canonical UTF-8 gloss. |
| class_index | integer | Stable contiguous zero-based training label. |
| view | enum | front, left, or right. |
| video_path | string | POSIX-style path relative to VSL400_ROOT. |
| metadata fields | nullable scalar | Values declared by release metadata. |
| measured media fields | nullable scalar | Values measured by ffprobe. |
| is_valid | boolean | False when any record-level error exists. |
| validation_errors | list of strings | Stable machine-readable error codes. |
| split | nullable enum | train, validation, or test. |

## Required invariants

1. sample_id is unique.
2. Each instance_id has exactly one front, one left, and one right sample.
3. All three samples in an instance have identical signer and gloss labels.
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

The default is deterministic 80/10/10 signer-disjoint splitting with seed 42.
For 28 signers, largest-remainder allocation yields 22 train, 3 validation, and
3 test signers. Multiple seeded candidates are scored using:

- deviation from target instance ratios;
- missing glosses per split;
- deviation from the global gloss distribution.

If the dataset release provides an authoritative signer allocation, place it at
data/splits/vsl400_official.json; it is validated for overlap, missing, and
unknown signers before use.
