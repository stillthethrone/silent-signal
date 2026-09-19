"""Explicit RTMPose-L 384x288 + RTMDet offline whole-body extraction."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from numpy.typing import NDArray

from silent_signal.pose.cache import (
    canonical_sha256,
    runtime_environment,
    sha256_file,
)
from silent_signal.pose.interface import (
    COCO_WHOLEBODY_KEYPOINTS,
    PoseCandidate,
    PoseExtractionError,
    RawPoseSequence,
)

_ENV_TOKEN = re.compile(r"\$\{(?:oc\.env:)?([A-Za-z_][A-Za-z0-9_]*)\}")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RTMPoseConfigurationError(ValueError):
    """Raised when the extraction configuration is ambiguous or unsafe."""


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    """Local OpenMMLab config/checkpoint plus immutable remote provenance."""

    config_path: Path
    checkpoint_path: Path
    checkpoint_url: str
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.config_path.suffix.lower() != ".py":
            raise RTMPoseConfigurationError(
                f"OpenMMLab config must be an explicit .py file: {self.config_path}"
            )
        if self.checkpoint_path.suffix.lower() not in {".pth", ".pt"}:
            raise RTMPoseConfigurationError(
                f"Checkpoint must be an explicit local .pth/.pt file: {self.checkpoint_path}"
            )
        if not self.checkpoint_url.startswith("https://"):
            raise RTMPoseConfigurationError("checkpoint_url must be an HTTPS provenance URL.")
        if self.checkpoint_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            self.checkpoint_sha256
        ):
            raise RTMPoseConfigurationError(
                "checkpoint_sha256 must contain 64 lowercase hex digits."
            )

    def verify(self, *, label: str) -> dict[str, str]:
        """Require local artifacts, calculate full hashes, and check optional pins."""

        if not self.config_path.is_file():
            raise RTMPoseConfigurationError(f"{label} config not found: {self.config_path}")
        if not self.checkpoint_path.is_file():
            raise RTMPoseConfigurationError(
                f"{label} checkpoint not found: {self.checkpoint_path}. "
                f"Download it from {self.checkpoint_url}"
            )
        config_sha256 = sha256_file(self.config_path)
        checkpoint_sha256 = sha256_file(self.checkpoint_path)
        if self.checkpoint_sha256 is not None and checkpoint_sha256 != self.checkpoint_sha256:
            raise RTMPoseConfigurationError(
                f"{label} checkpoint SHA-256 mismatch: expected {self.checkpoint_sha256}, "
                f"found {checkpoint_sha256}."
            )
        return {
            "config_path": str(self.config_path),
            "config_sha256": config_sha256,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_url": self.checkpoint_url,
            "checkpoint_sha256": checkpoint_sha256,
        }


@dataclass(frozen=True, slots=True)
class PersonSelectionConfig:
    """Deterministic primary-signer selection and crop expansion."""

    bbox_threshold: float = 0.30
    bbox_expansion: float = 1.20
    detector_score_weight: float = 0.20
    center_weight: float = 0.20
    area_weight: float = 0.10
    continuity_iou_weight: float = 0.50

    def __post_init__(self) -> None:
        if not 0 <= self.bbox_threshold <= 1:
            raise RTMPoseConfigurationError("bbox_threshold must be between 0 and 1.")
        if self.bbox_expansion < 1:
            raise RTMPoseConfigurationError("bbox_expansion must be at least 1.0.")
        weights = (
            self.detector_score_weight,
            self.center_weight,
            self.area_weight,
            self.continuity_iou_weight,
        )
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise RTMPoseConfigurationError(
                "Primary-person selection weights must be non-negative with a positive sum."
            )


@dataclass(frozen=True, slots=True)
class RTMPoseWholeBodyConfig:
    """Validated configuration for one reproducible extractor identity."""

    name: str
    config_path: Path
    pose_model: ArtifactConfig
    detector: ArtifactConfig
    device: str = "cuda:0"
    raw_layout: str = "coco_wholebody_133"
    variant: str = "rtmpose-l"
    training_dataset: str = "coco-wholebody"
    input_size_wh: tuple[int, int] = (288, 384)
    expected_num_keypoints: int = COCO_WHOLEBODY_KEYPOINTS
    person_category_id: int = 0
    selection: PersonSelectionConfig = field(default_factory=PersonSelectionConfig)
    hash_source_video: bool = True

    def __post_init__(self) -> None:
        if self.variant != "rtmpose-l":
            raise RTMPoseConfigurationError("The production extractor must use RTMPose-L.")
        if self.input_size_wh != (288, 384):
            raise RTMPoseConfigurationError(
                "RTMPose-L WholeBody must use input_size [288, 384] (width, height)."
            )
        if self.training_dataset != "coco-wholebody":
            raise RTMPoseConfigurationError("The pose checkpoint must be COCO-WholeBody trained.")
        if self.raw_layout != "coco_wholebody_133":
            raise RTMPoseConfigurationError("Raw extraction must preserve coco_wholebody_133.")
        if self.expected_num_keypoints != COCO_WHOLEBODY_KEYPOINTS:
            raise RTMPoseConfigurationError("Whole-body extraction must return exactly 133 points.")
        model_filename = self.pose_model.config_path.name.lower()
        if "rtmpose-l" not in model_filename or "384x288" not in model_filename:
            raise RTMPoseConfigurationError(
                "pose_model.config must explicitly name RTMPose-L WholeBody 384x288; aliases "
                "such as 'wholebody' are not accepted."
            )

    def verified_provenance(
        self,
        *,
        require_pinned_checkpoints: bool = True,
    ) -> dict[str, Any]:
        """Verify all local artifacts and return the complete extractor identity."""

        if require_pinned_checkpoints and (
            self.pose_model.checkpoint_sha256 is None or self.detector.checkpoint_sha256 is None
        ):
            raise RTMPoseConfigurationError(
                "Production extraction requires full checkpoint_sha256 pins for both "
                "RTMPose-L and RTMDet. Run `ss-extract-pose verify`, copy the reported "
                "digests into the YAML, and retry."
            )
        provenance: dict[str, Any] = {
            "schema_version": 1,
            "name": self.name,
            "framework": "mmpose",
            "variant": self.variant,
            "training_dataset": self.training_dataset,
            "input_size_wh": list(self.input_size_wh),
            "raw_layout": self.raw_layout,
            "expected_num_keypoints": self.expected_num_keypoints,
            "device": self.device,
            "person_category_id": self.person_category_id,
            "selection": {
                "bbox_threshold": self.selection.bbox_threshold,
                "bbox_expansion": self.selection.bbox_expansion,
                "detector_score_weight": self.selection.detector_score_weight,
                "center_weight": self.selection.center_weight,
                "area_weight": self.selection.area_weight,
                "continuity_iou_weight": self.selection.continuity_iou_weight,
            },
            "hash_source_video": self.hash_source_video,
            "extractor_config_path": str(self.config_path),
            "extractor_config_sha256": sha256_file(self.config_path),
            "pose_model": self.pose_model.verify(label="RTMPose-L"),
            "detector": self.detector.verify(label="RTMDet"),
            "runtime_environment": runtime_environment(),
        }
        return provenance


@dataclass(frozen=True, slots=True)
class DetectionCandidate:
    """One person detection before top-down pose estimation."""

    bbox_xyxy: NDArray[np.float32]
    score: float

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox_xyxy, dtype=np.float32)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError("Detection bbox must be a finite (4,) xyxy array.")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("Detection bbox must have positive width and height.")
        if not math.isfinite(self.score) or self.score < 0:
            raise ValueError("Detection score must be finite and non-negative.")
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "score", float(self.score))


class _Runtime(Protocol):
    def detect_people(self, frame: NDArray[np.uint8]) -> tuple[DetectionCandidate, ...]: ...

    def estimate_pose(
        self,
        frame: NDArray[np.uint8],
        bbox_xyxy: NDArray[np.float32],
        *,
        bbox_score: float,
    ) -> PoseCandidate: ...


def load_rtmpose_config(path: str | Path) -> RTMPoseWholeBodyConfig:
    """Load schema-versioned YAML while resolving explicit environment tokens."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RTMPoseConfigurationError(f"Pose configuration not found: {source}") from exc
    except yaml.YAMLError as exc:
        raise RTMPoseConfigurationError(f"Invalid pose YAML {source}: {exc}") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise RTMPoseConfigurationError("Pose configuration must use schema_version: 1.")
    extractor = _mapping(raw, "extractor")
    if extractor.get("framework") != "mmpose":
        raise RTMPoseConfigurationError("extractor.framework must be 'mmpose'.")
    model = _artifact(_mapping(extractor, "pose_model"), source.parent)
    detector = _artifact(_mapping(extractor, "detector"), source.parent)
    selection_raw = _optional_mapping(extractor, "person_selection")
    input_size = extractor.get("input_size_wh", [288, 384])
    if not isinstance(input_size, Sequence) or isinstance(input_size, (str, bytes)):
        raise RTMPoseConfigurationError("input_size_wh must be [width, height].")
    try:
        input_size_wh = tuple(int(value) for value in input_size)
    except (TypeError, ValueError) as exc:
        raise RTMPoseConfigurationError("input_size_wh must contain integers.") from exc
    if len(input_size_wh) != 2:
        raise RTMPoseConfigurationError("input_size_wh must contain exactly two integers.")
    return RTMPoseWholeBodyConfig(
        name=_required_str(extractor, "name"),
        config_path=source,
        pose_model=model,
        detector=detector,
        device=str(extractor.get("device", "cuda:0")),
        raw_layout=str(extractor.get("raw_layout", "coco_wholebody_133")),
        variant=str(extractor.get("variant", "rtmpose-l")),
        training_dataset=str(extractor.get("training_dataset", "coco-wholebody")),
        input_size_wh=(input_size_wh[0], input_size_wh[1]),
        expected_num_keypoints=int(
            extractor.get("expected_num_keypoints", COCO_WHOLEBODY_KEYPOINTS)
        ),
        person_category_id=int(extractor.get("person_category_id", 0)),
        selection=PersonSelectionConfig(
            bbox_threshold=float(selection_raw.get("bbox_threshold", 0.30)),
            bbox_expansion=float(selection_raw.get("bbox_expansion", 1.20)),
            detector_score_weight=float(selection_raw.get("detector_score_weight", 0.20)),
            center_weight=float(selection_raw.get("center_weight", 0.20)),
            area_weight=float(selection_raw.get("area_weight", 0.10)),
            continuity_iou_weight=float(selection_raw.get("continuity_iou_weight", 0.50)),
        ),
        hash_source_video=bool(extractor.get("hash_source_video", True)),
    )


class RTMPoseWholeBodyExtractor:
    """Decode a video and extract one stable signer using explicit OpenMMLab models."""

    def __init__(
        self,
        config: RTMPoseWholeBodyConfig,
        *,
        runtime: _Runtime | None = None,
        cv2_module: Any | None = None,
    ) -> None:
        self.config = config
        self._provenance = config.verified_provenance()
        self._fingerprint = canonical_sha256(_fingerprint_payload(self._provenance))
        self._runtime = runtime
        self._cv2 = cv2_module

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def extract_video(self, video_path: Path, *, sample_id: str) -> RawPoseSequence:
        """Run RTMDet and RTMPose-L on every decoded source frame."""

        source = Path(video_path).expanduser().resolve()
        if not source.is_file():
            raise PoseExtractionError(f"Video not found: {source}")
        cv2 = self._cv2 or _import_cv2()
        runtime = self._runtime or _OpenMMLabRuntime(self.config)
        self._runtime = runtime
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise PoseExtractionError(f"OpenCV cannot open video: {source}")

        fps = _positive_float(capture.get(cv2.CAP_PROP_FPS))
        frame_indices: list[int] = []
        timestamps: list[float] = []
        keypoints: list[NDArray[np.float32]] = []
        keypoint_scores: list[NDArray[np.float32]] = []
        bboxes: list[NDArray[np.float32]] = []
        bbox_scores: list[float] = []
        detected: list[bool] = []
        frame_size: tuple[int, int] | None = None
        previous_bbox: NDArray[np.float32] | None = None
        timestamp_fallbacks = 0

        try:
            frame_index = 0
            previous_timestamp = -1.0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                    raise PoseExtractionError(
                        f"Decoder returned an invalid frame at {frame_index}."
                    )
                height, width = int(frame.shape[0]), int(frame.shape[1])
                current_size = (height, width)
                if frame_size is None:
                    frame_size = current_size
                elif current_size != frame_size:
                    raise PoseExtractionError(
                        f"Video resolution changes at frame {frame_index}: "
                        f"{frame_size} -> {current_size}."
                    )

                timestamp, used_fallback = _frame_timestamp(
                    capture,
                    cv2,
                    frame_index=frame_index,
                    fps=fps,
                    previous=previous_timestamp,
                )
                timestamp_fallbacks += int(used_fallback)
                detections = tuple(
                    candidate
                    for candidate in runtime.detect_people(frame)
                    if candidate.score >= self.config.selection.bbox_threshold
                )
                selected = select_primary_detection(
                    detections,
                    frame_size_hw=current_size,
                    previous_bbox=previous_bbox,
                    config=self.config.selection,
                )
                if selected is None:
                    keypoints.append(np.zeros((COCO_WHOLEBODY_KEYPOINTS, 2), dtype=np.float32))
                    keypoint_scores.append(np.zeros((COCO_WHOLEBODY_KEYPOINTS,), dtype=np.float32))
                    bboxes.append(np.zeros((4,), dtype=np.float32))
                    bbox_scores.append(0.0)
                    detected.append(False)
                else:
                    previous_bbox = selected.bbox_xyxy
                    expanded_bbox = expand_bbox(
                        selected.bbox_xyxy,
                        factor=self.config.selection.bbox_expansion,
                        frame_size_hw=current_size,
                    )
                    pose = runtime.estimate_pose(
                        frame,
                        expanded_bbox,
                        bbox_score=selected.score,
                    )
                    keypoints.append(pose.keypoints_xy)
                    keypoint_scores.append(pose.keypoint_scores)
                    bboxes.append(pose.bbox_xyxy)
                    bbox_scores.append(pose.bbox_score)
                    detected.append(True)

                frame_indices.append(frame_index)
                timestamps.append(timestamp)
                previous_timestamp = timestamp
                frame_index += 1
        finally:
            capture.release()

        if frame_size is None or not frame_indices:
            raise PoseExtractionError(f"Video contains no decodable frames: {source}")
        video_sha256 = sha256_file(source) if self.config.hash_source_video else None
        metadata = {
            "extractor_fingerprint": self.fingerprint,
            "extractor_provenance": self.provenance,
            "video_sha256": video_sha256,
            "video_file_size_bytes": source.stat().st_size,
            "source_fps": fps,
            "timestamp_fallback_frames": timestamp_fallbacks,
            "raw_layout": self.config.raw_layout,
            "person_selection": "detector_score+center+area+temporal_iou",
            "missing_person_frames": len(detected) - sum(detected),
        }
        return RawPoseSequence(
            sample_id=sample_id,
            source_video=str(source),
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            timestamps_seconds=np.asarray(timestamps, dtype=np.float64),
            frame_size_hw=frame_size,
            keypoints_xy=np.stack(keypoints),
            keypoint_scores=np.stack(keypoint_scores),
            bboxes_xyxy=np.stack(bboxes),
            bbox_scores=np.asarray(bbox_scores, dtype=np.float32),
            person_detected=np.asarray(detected, dtype=np.bool_),
            metadata=metadata,
        )


class _OpenMMLabRuntime:
    """Lazy adapter around MMDetection 3.x and MMPose 1.x APIs."""

    def __init__(self, config: RTMPoseWholeBodyConfig) -> None:
        # Colab exports its notebook-only inline backend to child processes. The
        # extraction CLI is headless, so importing OpenMMLab must not inherit it.
        os.environ["MPLBACKEND"] = "Agg"
        try:
            from mmdet.apis import inference_detector, init_detector
            from mmdet.utils import register_all_modules as register_mmdet_modules
            from mmengine import DefaultScope
            from mmpose.apis import inference_topdown, init_model
            from mmpose.utils import register_all_modules as register_mmpose_modules
        except ImportError as exc:
            raise PoseExtractionError(
                "RTMPose extraction requires a compatible OpenMMLab environment with "
                "MMPose 1.x, MMDetection 3.x, MMCV 2.x, MMEngine, PyTorch and OpenCV."
            ) from exc
        register_mmdet_modules(init_default_scope=False)
        register_mmpose_modules(init_default_scope=True)
        self._scope_context = DefaultScope.overwrite_default_scope
        self._inference_detector = inference_detector
        self._inference_topdown = inference_topdown
        try:
            self._detector = init_detector(
                str(config.detector.config_path),
                str(config.detector.checkpoint_path),
                device=config.device,
            )
            self._pose_model = init_model(
                str(config.pose_model.config_path),
                str(config.pose_model.checkpoint_path),
                device=config.device,
            )
        except Exception as exc:
            raise PoseExtractionError(f"Cannot initialize OpenMMLab models: {exc}") from exc
        self._person_category_id = config.person_category_id

    def detect_people(self, frame: NDArray[np.uint8]) -> tuple[DetectionCandidate, ...]:
        try:
            # MMPose inference leaves the process-wide default scope at
            # ``mmpose``. MMDetection 3.2 builds its test transforms lazily, so
            # restore ``mmdet`` before every frame or PackDetInputs is looked up
            # in the wrong registry.
            with self._scope_context("mmdet"):
                result = self._inference_detector(self._detector, frame)
            instances = result.pred_instances.cpu().numpy()
            bboxes = np.asarray(instances.bboxes, dtype=np.float32)
            scores = np.asarray(instances.scores, dtype=np.float32)
            labels = np.asarray(instances.labels, dtype=np.int64)
        except Exception as exc:
            raise PoseExtractionError(f"RTMDet inference failed: {exc}") from exc
        return tuple(
            DetectionCandidate(bbox, float(score))
            for bbox, score, label in zip(bboxes, scores, labels, strict=True)
            if int(label) == self._person_category_id
        )

    def estimate_pose(
        self,
        frame: NDArray[np.uint8],
        bbox_xyxy: NDArray[np.float32],
        *,
        bbox_score: float,
    ) -> PoseCandidate:
        try:
            with self._scope_context("mmpose"):
                results = self._inference_topdown(
                    self._pose_model,
                    frame,
                    np.asarray([bbox_xyxy], dtype=np.float32),
                )
            if len(results) != 1:
                raise PoseExtractionError(
                    f"RTMPose returned {len(results)} samples for one person bbox."
                )
            instances = results[0].pred_instances.cpu().numpy()
            keypoints = np.asarray(instances.keypoints, dtype=np.float32)
            scores = np.asarray(instances.keypoint_scores, dtype=np.float32)
        except PoseExtractionError:
            raise
        except Exception as exc:
            raise PoseExtractionError(f"RTMPose inference failed: {exc}") from exc
        if keypoints.shape == (1, COCO_WHOLEBODY_KEYPOINTS, 2):
            keypoints = keypoints[0]
        if scores.shape == (1, COCO_WHOLEBODY_KEYPOINTS):
            scores = scores[0]
        try:
            return PoseCandidate(
                keypoints_xy=keypoints,
                keypoint_scores=scores,
                bbox_xyxy=bbox_xyxy,
                bbox_score=bbox_score,
            )
        except ValueError as exc:
            raise PoseExtractionError(
                f"RTMPose produced an invalid 133-keypoint result: {exc}"
            ) from exc


def select_primary_detection(
    candidates: Sequence[DetectionCandidate],
    *,
    frame_size_hw: tuple[int, int],
    previous_bbox: NDArray[np.float32] | None,
    config: PersonSelectionConfig,
) -> DetectionCandidate | None:
    """Select the central signer while preserving temporal identity by IoU."""

    if not candidates:
        return None
    height, width = frame_size_hw
    if height <= 0 or width <= 0:
        raise ValueError("frame_size_hw must be positive.")
    frame_area = float(height * width)
    frame_center = np.asarray([width / 2, height / 2], dtype=np.float32)
    half_diagonal = max(math.hypot(width, height) / 2, 1.0)

    def rank(candidate: DetectionCandidate) -> tuple[float, float, float]:
        bbox = candidate.bbox_xyxy
        center = np.asarray([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        center_score = max(0.0, 1.0 - float(np.linalg.norm(center - frame_center)) / half_diagonal)
        area = max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        area_score = min(1.0, area / frame_area)
        continuity = bbox_iou(bbox, previous_bbox) if previous_bbox is not None else 0.0
        total = (
            config.detector_score_weight * min(candidate.score, 1.0)
            + config.center_weight * center_score
            + config.area_weight * area_score
            + config.continuity_iou_weight * continuity
        )
        return total, candidate.score, area

    return max(candidates, key=rank)


def expand_bbox(
    bbox_xyxy: NDArray[np.floating],
    *,
    factor: float,
    frame_size_hw: tuple[int, int],
) -> NDArray[np.float32]:
    """Expand an xyxy crop around its center and clip it to image boundaries."""

    if factor < 1:
        raise ValueError("BBox expansion factor must be at least 1.0.")
    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    if bbox.shape != (4,):
        raise ValueError("bbox_xyxy must have shape (4,).")
    height, width = frame_size_hw
    center_x = float(bbox[0] + bbox[2]) / 2
    center_y = float(bbox[1] + bbox[3]) / 2
    half_width = float(bbox[2] - bbox[0]) * factor / 2
    half_height = float(bbox[3] - bbox[1]) * factor / 2
    return np.asarray(
        [
            max(0.0, center_x - half_width),
            max(0.0, center_y - half_height),
            min(float(width - 1), center_x + half_width),
            min(float(height - 1), center_y + half_height),
        ],
        dtype=np.float32,
    )


def bbox_iou(
    first: NDArray[np.floating],
    second: NDArray[np.floating],
) -> float:
    """Calculate intersection-over-union for two xyxy boxes."""

    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _artifact(raw: Mapping[str, Any], base_dir: Path) -> ArtifactConfig:
    checksum = raw.get("checkpoint_sha256")
    checksum_value = None if checksum in (None, "") else str(checksum).lower()
    return ArtifactConfig(
        config_path=_resolve_path(_required_str(raw, "config"), base_dir),
        checkpoint_path=_resolve_path(_required_str(raw, "checkpoint"), base_dir),
        checkpoint_url=_required_str(raw, "checkpoint_url"),
        checkpoint_sha256=checksum_value,
    )


def _fingerprint_payload(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Remove machine-local paths while retaining every content and runtime identity."""

    payload = dict(provenance)
    payload.pop("extractor_config_path", None)
    for key in ("pose_model", "detector"):
        artifact = dict(payload[key])
        artifact.pop("config_path", None)
        artifact.pop("checkpoint_path", None)
        payload[key] = artifact
    return payload


def _resolve_path(value: str, base_dir: Path) -> Path:
    expanded = _ENV_TOKEN.sub(_replace_env_token, value)
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _replace_env_token(match: re.Match[str]) -> str:
    name = match.group(1)
    value = os.environ.get(name)
    if not value:
        raise RTMPoseConfigurationError(f"Environment variable {name} is not set.")
    return value


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RTMPoseConfigurationError(f"{key} must be a mapping.")
    return value


def _optional_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise RTMPoseConfigurationError(f"{key} must be a mapping.")
    return value


def _required_str(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RTMPoseConfigurationError(f"{key} must be a non-empty string.")
    return value.strip()


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _frame_timestamp(
    capture: Any,
    cv2: Any,
    *,
    frame_index: int,
    fps: float | None,
    previous: float,
) -> tuple[float, bool]:
    milliseconds = _positive_float(capture.get(cv2.CAP_PROP_POS_MSEC))
    candidate = milliseconds / 1000 if milliseconds is not None else None
    if candidate is not None and candidate > previous:
        return candidate, False
    if fps is not None:
        return frame_index / fps, True
    return (0.0 if frame_index == 0 else previous + 1 / 30), True


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise PoseExtractionError(
            "Video decoding requires OpenCV. Install the documented pose environment."
        ) from exc
    return cv2
