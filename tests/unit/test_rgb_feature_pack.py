from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from silent_signal.cli.extract_videomae_features import (
    _processor_pixel_values,
    extraction_fingerprint,
    midpoint_frame_indices,
)
from silent_signal.data.rgb_feature_pack import (
    RGBFeaturePackError,
    read_rgb_feature_pack,
    write_rgb_feature_pack,
)


def test_rgb_feature_pack_round_trip_is_pickle_free_and_indexed(tmp_path: Path) -> None:
    path = tmp_path / "rgb.npz"
    features = np.arange(3 * 8 * 768, dtype=np.float32).reshape(3, 8, 768)
    metadata = {"extraction_fingerprint": "a" * 64, "model": "test"}

    write_rgb_feature_pack(path, ["sample-c", "sample-a", "sample-b"], features, metadata)
    packed = read_rgb_feature_pack(path, expected_fingerprint="a" * 64)

    assert packed.sample_ids == ("sample-c", "sample-a", "sample-b")
    assert packed.features.shape == (3, 8, 768)
    assert packed.features.dtype == np.float16
    assert packed.metadata == metadata
    assert packed.index == {"sample-c": 0, "sample-a": 1, "sample-b": 2}


def test_rgb_feature_pack_rejects_wrong_fingerprint_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "rgb.npz"
    features = np.zeros((2, 8, 768), dtype=np.float16)

    with pytest.raises(RGBFeaturePackError, match="unique"):
        write_rgb_feature_pack(path, ["same", "same"], features, {})

    write_rgb_feature_pack(
        path,
        ["first", "second"],
        features,
        {"extraction_fingerprint": "correct"},
    )
    with pytest.raises(RGBFeaturePackError, match="fingerprint mismatch"):
        read_rgb_feature_pack(path, expected_fingerprint="wrong")


def test_rgb_feature_pack_rejects_non_object_metadata(tmp_path: Path) -> None:
    path = tmp_path / "invalid-metadata.npz"
    envelope = json.dumps({"schema_version": 1, "metadata": ["not", "an", "object"]})
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["sample"]),
        features=np.zeros((1, 8, 768), dtype=np.float16),
        metadata_json=np.frombuffer(envelope.encode("utf-8"), dtype=np.uint8),
    )

    with pytest.raises(RGBFeaturePackError, match="metadata must be a JSON object"):
        read_rgb_feature_pack(path)


def test_extraction_fingerprint_pins_manifest_model_and_preprocessing() -> None:
    first = extraction_fingerprint("manifest-a")

    assert first == extraction_fingerprint("manifest-a")
    assert first != extraction_fingerprint("manifest-b")
    assert first != extraction_fingerprint("manifest-a", model_revision="different")


def test_midpoint_sampler_is_deterministic_and_handles_short_clips() -> None:
    assert midpoint_frame_indices(32, frames=4).tolist() == [4, 12, 20, 28]
    assert midpoint_frame_indices(1).tolist() == [0] * 16
    with pytest.raises(ValueError, match="positive"):
        midpoint_frame_indices(0)


class _FakeProcessor:
    def __init__(self, *, already_channel_first: bool = False) -> None:
        self.already_channel_first = already_channel_first

    def __call__(self, frames: list[np.ndarray], *, return_tensors: str):
        assert len(frames) == 16
        assert return_tensors == "pt"
        values = torch.arange(16 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 16, 3, 2, 2)
        if self.already_channel_first:
            values = values.permute(0, 2, 1, 3, 4)
        return {"pixel_values": values}


@pytest.mark.parametrize("already_channel_first", [False, True])
def test_processor_helper_converts_official_layout_to_model_layout(
    already_channel_first: bool,
) -> None:
    video = np.zeros((16, 2, 2, 3), dtype=np.uint8)

    values = _processor_pixel_values(
        _FakeProcessor(already_channel_first=already_channel_first),
        [video, video],
        torch,
    )

    assert values.shape == (2, 3, 16, 2, 2)
    assert torch.equal(values[0], values[1])
