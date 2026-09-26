"""Extract resumable frozen VideoMAE V2 temporal-token caches from local videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from silent_signal.contracts import ManifestRecord
from silent_signal.data.manifest import read_manifest
from silent_signal.data.rgb_feature_pack import (
    RGBFeaturePackError,
    read_rgb_feature_pack,
    write_rgb_feature_pack,
)
from silent_signal.pose.cache import canonical_sha256, sha256_file, write_json_atomic

DEFAULT_MODEL_ID = "OpenGVLab/VideoMAEv2-Base"
DEFAULT_MODEL_REVISION = "0e826d7e85e39f9d951e331cd91c5c2d8142d385"
INPUT_FRAMES = 16
EXPECTED_TEMPORAL_TOKENS = 8
EXPECTED_FEATURE_DIM = 768
PREPROCESSING = {
    "frame_sampler": "uniform_bin_midpoint_v1",
    "frames": INPUT_FRAMES,
    "image_processor": "VideoMAEImageProcessor.from_pretrained",
    "processor_layout": "B,T,C,H,W",
    "model_layout": "B,C,T,H,W",
    "token_reduction": "spatial_patch_mean_after_frozen_backbone",
    "temporal_tokens": EXPECTED_TEMPORAL_TOKENS,
    "feature_dim": EXPECTED_FEATURE_DIM,
    "dtype": "float16",
}

ExtractBatch = Callable[[Sequence[Path]], NDArray[np.generic]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m silent_signal.cli.extract_videomae_features",
        description="Cache frozen VideoMAE V2 temporal tokens in resumable atomic shards.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def extraction_fingerprint(
    manifest_sha256: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
) -> str:
    """Return the immutable identity of manifest, model and RGB preprocessing."""

    return canonical_sha256(
        {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "model_id": model_id,
            "model_revision": model_revision,
            "preprocessing": PREPROCESSING,
        }
    )


def midpoint_frame_indices(frame_count: int, frames: int = INPUT_FRAMES) -> NDArray[np.int64]:
    """Choose one deterministic midpoint from each of ``frames`` temporal bins."""

    if frame_count < 1 or frames < 1:
        raise ValueError("frame_count and frames must be positive.")
    edges = np.linspace(0.0, float(frame_count), frames + 1, dtype=np.float64)
    indices = ((edges[:-1] + edges[1:]) * 0.5).astype(np.int64)
    return np.minimum(indices, frame_count - 1)


def extract_videomae_features(
    *,
    manifest_path: Path,
    dataset_root: Path,
    output_path: Path,
    report_path: Path,
    shard_root: Path,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    batch_size: int = 2,
    shard_size: int = 256,
    progress_every: int = 10,
    device_name: str = "auto",
    overwrite: bool = False,
    extract_batch: ExtractBatch | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Create or reuse a complete feature pack, resuming any valid shard files."""

    if min(batch_size, shard_size) < 1 or progress_every < 0:
        raise ValueError("batch-size and shard-size must be positive; progress-every >= 0.")
    started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    shard_root = shard_root.resolve()
    manifest_sha256 = sha256_file(manifest_path)
    fingerprint = extraction_fingerprint(
        manifest_sha256,
        model_id=model_id,
        model_revision=model_revision,
    )
    records = _validated_records(manifest_path)
    sample_ids = tuple(record.sample_id for record in records)

    # This check deliberately happens before any MP4 validation or heavyweight import.
    # A completed Drive cache must be reusable after Colab's temporary videos disappear.
    if not overwrite and completed_output_is_current(
        output_path,
        report_path,
        sample_ids=sample_ids,
        extraction_identity=fingerprint,
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        log(f"[reuse] complete RGB feature pack: {output_path}")
        return report

    common_metadata = {
        "extraction_fingerprint": fingerprint,
        "manifest_sha256": manifest_sha256,
        "model_id": model_id,
        "model_revision": model_revision,
        "preprocessing": PREPROCESSING,
    }
    shard_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total_shards = (len(records) + shard_size - 1) // shard_size
    resumed_shards = 0
    extracted_shards = 0
    runtime_extract = extract_batch

    for shard_index in range(total_shards):
        start = shard_index * shard_size
        stop = min(len(records), start + shard_size)
        shard_records = records[start:stop]
        shard_ids = tuple(record.sample_id for record in shard_records)
        shard_path = shard_root / f"shard_{shard_index:05d}.npz"
        shard_metadata = {
            **common_metadata,
            "kind": "videomae_temporal_token_shard",
            "shard_index": shard_index,
            "start": start,
            "stop": stop,
            "samples": len(shard_records),
        }
        if not overwrite and _shard_is_current(
            shard_path,
            shard_ids=shard_ids,
            metadata=shard_metadata,
        ):
            resumed_shards += 1
            log(f"[shard {shard_index + 1}/{total_shards}] reuse {len(shard_ids)} samples")
            continue

        paths = [_safe_rgb_path(dataset_root, record) for record in shard_records]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} RGB videos are missing for shard {shard_index}; first: "
                f"{missing[0]}"
            )
        if runtime_extract is None:
            runtime = _VideoMAERuntime(
                model_id=model_id,
                model_revision=model_revision,
                device_name=device_name,
            )
            runtime_extract = runtime.extract
        features = _extract_paths(
            paths,
            runtime_extract,
            batch_size=batch_size,
            progress_every=progress_every,
            label=f"shard {shard_index + 1}/{total_shards}",
            log=log,
        )
        _validate_feature_shape(features, len(shard_ids))
        write_rgb_feature_pack(
            shard_path,
            shard_ids,
            features,
            shard_metadata,
            overwrite=shard_path.exists(),
        )
        extracted_shards += 1
        log(f"[shard {shard_index + 1}/{total_shards}] saved: {shard_path}")

    final_features = np.empty(
        (len(records), EXPECTED_TEMPORAL_TOKENS, EXPECTED_FEATURE_DIM), dtype=np.float16
    )
    for shard_index in range(total_shards):
        start = shard_index * shard_size
        stop = min(len(records), start + shard_size)
        shard_path = shard_root / f"shard_{shard_index:05d}.npz"
        shard = read_rgb_feature_pack(shard_path, expected_fingerprint=fingerprint)
        expected_ids = sample_ids[start:stop]
        if shard.sample_ids != expected_ids:
            raise RGBFeaturePackError(f"Shard sample order mismatch: {shard_path}")
        _validate_feature_shape(shard.features, len(expected_ids))
        final_features[start:stop] = shard.features

    final_metadata = {
        **common_metadata,
        "kind": "videomae_temporal_token_pack",
        "samples": len(records),
        "feature_shape": [EXPECTED_TEMPORAL_TOKENS, EXPECTED_FEATURE_DIM],
        "shards": total_shards,
    }
    write_rgb_feature_pack(
        output_path,
        sample_ids,
        final_features,
        final_metadata,
        overwrite=output_path.exists(),
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        **common_metadata,
        "manifest": str(manifest_path),
        "dataset_root": str(dataset_root.resolve()),
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "samples": len(records),
        "classes": len({record.class_index for record in records}),
        "splits": dict(Counter(str(record.split) for record in records)),
        "feature_shape": [len(records), EXPECTED_TEMPORAL_TOKENS, EXPECTED_FEATURE_DIM],
        "dtype": "float16",
        "shard_root": str(shard_root),
        "shards": total_shards,
        "resumed_shards": resumed_shards,
        "extracted_shards": extracted_shards,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    write_json_atomic(report_path, report)
    log(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _validated_records(manifest_path: Path) -> tuple[ManifestRecord, ...]:
    records = tuple(sorted(read_manifest(manifest_path), key=lambda record: record.sample_id))
    if not records:
        raise ValueError("Manifest contains no records.")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest sample IDs must be unique.")
    if any(record.split not in {"train", "validation", "test"} for record in records):
        raise ValueError("Every manifest record needs a train, validation or test split.")
    classes = sorted({record.class_index for record in records})
    if classes != list(range(len(classes))):
        raise ValueError("Manifest class indices must be contiguous from zero.")
    return records


def _safe_rgb_path(dataset_root: Path, record: ManifestRecord) -> Path:
    relative = PurePosixPath(record.video_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest video path: {record.video_path}")
    if relative.suffix.lower() not in {".npy", ".mp4"}:
        raise ValueError(f"Cannot derive RGB path from {record.video_path!r}.")
    rgb_relative = relative.with_suffix(".mp4")
    root = dataset_root.resolve()
    candidate = root.joinpath(*rgb_relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"RGB path escapes dataset root: {record.video_path}") from exc
    return candidate


def completed_output_is_current(
    output_path: Path,
    report_path: Path,
    *,
    sample_ids: tuple[str, ...],
    extraction_identity: str,
) -> bool:
    """Return whether the final pack and report exactly match this extraction identity.

    This lightweight public check intentionally needs no videos and imports no model
    runtime, so a Colab notebook can decide whether the temporary MP4 tree is needed.
    """
    if not output_path.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "complete"
            or report.get("extraction_fingerprint") != extraction_identity
            or int(report.get("samples", -1)) != len(sample_ids)
            or int(report.get("output_bytes", -1)) != output_path.stat().st_size
            or report.get("output_sha256") != sha256_file(output_path)
        ):
            return False
        with np.load(output_path, allow_pickle=False) as archive:
            envelope = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
            metadata = dict(envelope.get("metadata", {}))
            packed_ids = tuple(str(value) for value in archive["sample_ids"])
            feature_shape = archive["features"].shape
            feature_dtype = archive["features"].dtype
        return (
            metadata.get("extraction_fingerprint") == extraction_identity
            and packed_ids == sample_ids
            and feature_shape
            == (len(sample_ids), EXPECTED_TEMPORAL_TOKENS, EXPECTED_FEATURE_DIM)
            and feature_dtype == np.float16
        )
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _shard_is_current(
    path: Path,
    *,
    shard_ids: tuple[str, ...],
    metadata: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        pack = read_rgb_feature_pack(
            path,
            expected_fingerprint=str(metadata["extraction_fingerprint"]),
        )
        _validate_feature_shape(pack.features, len(shard_ids))
        return pack.sample_ids == shard_ids and pack.metadata == metadata
    except (RGBFeaturePackError, ValueError):
        return False


def _extract_paths(
    paths: Sequence[Path],
    extract_batch: ExtractBatch,
    *,
    batch_size: int,
    progress_every: int,
    label: str,
    log: Callable[[str], None],
) -> NDArray[np.float16]:
    chunks: list[NDArray[np.float16]] = []
    total_batches = (len(paths) + batch_size - 1) // batch_size
    started = time.perf_counter()
    for batch_index, start in enumerate(range(0, len(paths), batch_size), start=1):
        values = np.asarray(extract_batch(paths[start : start + batch_size]), dtype=np.float16)
        _validate_feature_shape(values, min(batch_size, len(paths) - start))
        chunks.append(values)
        if progress_every and (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % progress_every == 0
        ):
            elapsed = time.perf_counter() - started
            rate = batch_index / elapsed if elapsed else 0.0
            eta = (total_batches - batch_index) / rate / 60 if rate else 0.0
            log(f"[{label}] batch {batch_index}/{total_batches} | ETA {eta:.1f} min")
    return np.concatenate(chunks, axis=0)


def _validate_feature_shape(features: NDArray[np.generic], samples: int) -> None:
    expected = (samples, EXPECTED_TEMPORAL_TOKENS, EXPECTED_FEATURE_DIM)
    if features.shape != expected:
        raise ValueError(f"Expected VideoMAE features {expected}, found {features.shape}.")
    if not np.isfinite(features).all():
        raise ValueError("VideoMAE features contain non-finite values.")


class _VideoMAERuntime:
    """Lazy heavyweight runtime used only while at least one shard is missing."""

    def __init__(self, *, model_id: str, model_revision: str, device_name: str) -> None:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("USE_JAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        import cv2
        import torch
        from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

        selected = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
        if selected == "auto":
            selected = "cpu"
        if selected == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.cv2 = cv2
        self.torch = torch
        self.device = torch.device(selected)
        self.processor = VideoMAEImageProcessor.from_pretrained(
            model_id,
            revision=model_revision,
        )
        config = AutoConfig.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=model_revision,
            config=config,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def extract(self, paths: Sequence[Path]) -> NDArray[np.float16]:
        videos = [_decode_midpoint_frames(path, self.cv2) for path in paths]
        pixel_values = _processor_pixel_values(self.processor, videos, self.torch).to(self.device)
        with (
            self.torch.inference_mode(),
            self.torch.autocast(
                device_type=self.device.type,
                dtype=self.torch.float16,
                enabled=self.device.type == "cuda",
            ),
        ):
            tokens = _frozen_temporal_tokens(self.model, pixel_values)
        return tokens.float().cpu().numpy().astype(np.float16)


def _decode_midpoint_frames(path: Path, cv2: Any) -> NDArray[np.uint8]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        capture.release()
        raise RuntimeError(f"Video has no frames: {path}")
    targets = midpoint_frame_indices(frame_count)
    target_set = set(int(value) for value in targets)
    decoded: dict[int, NDArray[np.uint8]] = {}
    position = 0
    while position <= int(targets[-1]):
        ok, frame = capture.read()
        if not ok:
            break
        if position in target_set:
            decoded[position] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        position += 1
    capture.release()
    if not decoded:
        raise RuntimeError(f"Failed to decode frames: {path}")
    available = sorted(decoded)
    frames = [
        decoded[min(available, key=lambda value: abs(value - int(target)))] for target in targets
    ]
    return np.stack(frames)


def _processor_pixel_values(processor: Any, videos: Sequence[NDArray[np.uint8]], torch: Any):
    batches = []
    for video in videos:
        output = processor(list(video), return_tensors="pt")
        values = output["pixel_values"]
        if values.ndim != 5 or values.shape[0] != 1:
            raise RuntimeError(
                f"VideoMAEImageProcessor returned unexpected shape {tuple(values.shape)}."
            )
        if values.shape[1] == INPUT_FRAMES and values.shape[2] == 3:
            values = values.permute(0, 2, 1, 3, 4)
        elif values.shape[1] != 3 or values.shape[2] != INPUT_FRAMES:
            raise RuntimeError(
                f"Cannot convert processor output {tuple(values.shape)} to [B,3,16,H,W]."
            )
        batches.append(values.contiguous())
    return torch.cat(batches, dim=0)


def _frozen_temporal_tokens(model: Any, pixel_values: Any):
    visual = model.model
    tokens = visual.patch_embed(pixel_values)
    if visual.pos_embed is not None:
        position = visual.pos_embed.expand(tokens.size(0), -1, -1)
        tokens = tokens + position.type_as(tokens).to(tokens.device).detach()
    tokens = visual.pos_drop(tokens)
    for block in visual.blocks:
        tokens = block(tokens)
    temporal = pixel_values.shape[2] // int(visual.tubelet_size)
    if tokens.shape[1] % temporal:
        raise RuntimeError("VideoMAE token count is not divisible by temporal tubes.")
    spatial = tokens.shape[1] // temporal
    tokens = tokens.reshape(tokens.shape[0], temporal, spatial, tokens.shape[2]).mean(dim=2)
    return visual.fc_norm(tokens) if visual.fc_norm is not None else visual.norm(tokens)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_root = args.shard_root or args.output.with_suffix("").with_name(
        args.output.stem + "_shards"
    )
    try:
        extract_videomae_features(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            output_path=args.output,
            report_path=args.report,
            shard_root=shard_root,
            model_id=args.model_id,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            progress_every=args.progress_every,
            device_name=args.device,
            overwrite=args.overwrite,
        )
        return 0
    except (OSError, RGBFeaturePackError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
