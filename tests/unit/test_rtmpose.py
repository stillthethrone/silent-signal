from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from silent_signal.pose.interface import PoseCandidate
from silent_signal.pose.rtmpose import (
    ArtifactConfig,
    DetectionCandidate,
    PersonSelectionConfig,
    RTMPoseConfigurationError,
    RTMPoseWholeBodyConfig,
    RTMPoseWholeBodyExtractor,
    bbox_iou,
    expand_bbox,
    load_rtmpose_config,
    select_primary_detection,
)


def _artifact(tmp_path: Path, prefix: str, config_name: str) -> ArtifactConfig:
    config = tmp_path / config_name
    checkpoint = tmp_path / f"{prefix}.pth"
    config.write_text("model = dict()\n", encoding="utf-8")
    checkpoint.write_bytes(prefix.encode("ascii"))
    return ArtifactConfig(
        config_path=config,
        checkpoint_path=checkpoint,
        checkpoint_url=f"https://example.test/{prefix}.pth",
        checkpoint_sha256=hashlib.sha256(prefix.encode("ascii")).hexdigest(),
    )


def _config(tmp_path: Path) -> RTMPoseWholeBodyConfig:
    source = tmp_path / "extractor.yaml"
    source.write_text("schema_version: 1\n", encoding="utf-8")
    return RTMPoseWholeBodyConfig(
        name="rtmpose_l_coco_wholebody_384x288",
        config_path=source,
        pose_model=_artifact(
            tmp_path,
            "pose",
            "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py",
        ),
        detector=_artifact(tmp_path, "detector", "rtmdet_m_coco-person.py"),
    )


def test_config_loader_requires_explicit_model_paths_and_resolves_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mmpose_root = tmp_path / "mmpose"
    pose_config = mmpose_root / "rtmpose-l_8xb32-270e_coco-wholebody-384x288.py"
    detector_config = mmpose_root / "rtmdet_m_coco-person.py"
    pose_checkpoint = tmp_path / "pose.pth"
    detector_checkpoint = tmp_path / "detector.pth"
    for path in (pose_config, detector_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model = dict()\n", encoding="utf-8")
    pose_checkpoint.write_bytes(b"pose")
    detector_checkpoint.write_bytes(b"detector")
    monkeypatch.setenv("MMPOSE_ROOT", str(mmpose_root))
    monkeypatch.setenv("POSE_CKPT", str(pose_checkpoint))
    monkeypatch.setenv("DET_CKPT", str(detector_checkpoint))
    path = tmp_path / "rtmpose.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "extractor": {
                    "name": "rtmpose_l_coco_wholebody_384x288",
                    "framework": "mmpose",
                    "variant": "rtmpose-l",
                    "training_dataset": "coco-wholebody",
                    "raw_layout": "coco_wholebody_133",
                    "input_size_wh": [288, 384],
                    "pose_model": {
                        "config": "${MMPOSE_ROOT}/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py",
                        "checkpoint": "${POSE_CKPT}",
                        "checkpoint_url": "https://example.test/pose.pth",
                    },
                    "detector": {
                        "config": "${MMPOSE_ROOT}/rtmdet_m_coco-person.py",
                        "checkpoint": "${DET_CKPT}",
                        "checkpoint_url": "https://example.test/detector.pth",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_rtmpose_config(path)
    provenance = config.verified_provenance(require_pinned_checkpoints=False)

    assert config.pose_model.config_path == pose_config
    assert len(provenance["pose_model"]["checkpoint_sha256"]) == 64
    assert len(provenance["detector"]["checkpoint_sha256"]) == 64


def test_config_rejects_mutable_alias_instead_of_rtmpose_l_file(tmp_path: Path) -> None:
    with pytest.raises(RTMPoseConfigurationError, match=r"explicit \.py"):
        ArtifactConfig(
            config_path=Path("wholebody"),
            checkpoint_path=tmp_path / "pose.pth",
            checkpoint_url="https://example.test/pose.pth",
        )


def test_production_extractor_requires_full_checkpoint_pins(tmp_path: Path) -> None:
    config = _config(tmp_path)
    unpinned_pose = ArtifactConfig(
        config_path=config.pose_model.config_path,
        checkpoint_path=config.pose_model.checkpoint_path,
        checkpoint_url=config.pose_model.checkpoint_url,
    )
    unpinned = RTMPoseWholeBodyConfig(
        name=config.name,
        config_path=config.config_path,
        pose_model=unpinned_pose,
        detector=config.detector,
    )

    with pytest.raises(RTMPoseConfigurationError, match="requires full checkpoint_sha256"):
        RTMPoseWholeBodyExtractor(unpinned, runtime=_FakeRuntime(), cv2_module=_FakeCv2([]))


def test_primary_detection_prefers_temporal_signer_and_expands_safely() -> None:
    previous = np.asarray([20, 20, 80, 180], dtype=np.float32)
    continuing = DetectionCandidate(np.asarray([22, 22, 82, 182]), 0.8)
    distractor = DetectionCandidate(np.asarray([100, 10, 195, 195]), 0.99)

    selected = select_primary_detection(
        (continuing, distractor),
        frame_size_hw=(200, 200),
        previous_bbox=previous,
        config=PersonSelectionConfig(),
    )
    expanded = expand_bbox(
        np.asarray([-10, 10, 190, 210]),
        factor=1.2,
        frame_size_hw=(200, 200),
    )

    assert selected is continuing
    assert bbox_iou(previous, continuing.bbox_xyxy) > 0.8
    np.testing.assert_array_equal(expanded, np.asarray([0, 0, 199, 199], dtype=np.float32))


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], cv2: Any) -> None:
        self.frames = frames
        self.cv2 = cv2
        self.position = 0

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def get(self, prop: int) -> float:
        if prop == self.cv2.CAP_PROP_FPS:
            return 25.0
        if prop == self.cv2.CAP_PROP_POS_MSEC:
            return max(0, self.position - 1) * 40.0
        return 0.0

    def release(self) -> None:
        return None


class _FakeCv2:
    CAP_PROP_FPS = 1
    CAP_PROP_POS_MSEC = 2

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames

    def VideoCapture(self, _path: str) -> _FakeCapture:
        return _FakeCapture(self.frames, self)


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def detect_people(self, _frame: np.ndarray) -> tuple[DetectionCandidate, ...]:
        self.calls += 1
        if self.calls == 2:
            return ()
        return (DetectionCandidate(np.asarray([10, 10, 80, 90]), 0.95),)

    def estimate_pose(
        self,
        _frame: np.ndarray,
        bbox_xyxy: np.ndarray,
        *,
        bbox_score: float,
    ) -> PoseCandidate:
        return PoseCandidate(
            keypoints_xy=np.ones((133, 2), dtype=np.float32),
            keypoint_scores=np.full((133,), 0.8, dtype=np.float32),
            bbox_xyxy=bbox_xyxy,
            bbox_score=bbox_score,
        )


def test_extractor_preserves_all_frames_and_marks_missing_person(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic video identity")
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    extractor = RTMPoseWholeBodyExtractor(
        _config(tmp_path),
        runtime=_FakeRuntime(),
        cv2_module=_FakeCv2(frames),
    )

    result = extractor.extract_video(video, sample_id="sample-1")

    assert result.keypoints_xy.shape == (3, 133, 2)
    assert result.person_detected.tolist() == [True, False, True]
    assert not result.keypoints_xy[1].any()
    assert result.timestamps_seconds.tolist() == pytest.approx([0.0, 0.04, 0.08])
    assert result.metadata["extractor_fingerprint"] == extractor.fingerprint
    assert result.metadata["missing_person_frames"] == 1
