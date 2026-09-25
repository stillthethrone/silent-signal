from __future__ import annotations

import numpy as np

from silent_signal.pose.layouts import COCO_WHOLEBODY_75_V1


def test_layout_is_a_stable_75_joint_coco_wholebody_subset() -> None:
    layout = COCO_WHOLEBODY_75_V1

    assert layout.name == "coco_wholebody_75_v1"
    assert layout.source_layout == "coco_wholebody_133"
    assert layout.num_joints == 75
    assert len(set(layout.source_indices)) == 75
    assert layout.source_indices[:13] == tuple(range(13))
    assert layout.source_indices[13:34] == tuple(range(91, 112))
    assert layout.source_indices[34:55] == tuple(range(112, 133))
    assert all(0 <= start < 75 and 0 <= end < 75 for start, end in layout.edges)
    assert (9, 13) in layout.edges
    assert (10, 34) in layout.edges


def test_layout_selects_matching_coordinates_and_scores() -> None:
    layout = COCO_WHOLEBODY_75_V1
    xy = np.arange(2 * 133 * 2, dtype=np.float32).reshape(2, 133, 2)
    scores = np.arange(2 * 133, dtype=np.float32).reshape(2, 133)

    selected_xy, selected_scores = layout.select(xy, scores)

    assert selected_xy.shape == (2, 75, 2)
    assert selected_scores.shape == (2, 75)
    np.testing.assert_array_equal(selected_xy[:, 13], xy[:, 91])
    np.testing.assert_array_equal(selected_scores[:, 34], scores[:, 112])


# Copied from nguyenanfms/VSL-VietnameseSignLanguage src/features/normalizer.py and
# keypoints.py, which wrote the Kaggle [T, 76, 3] arrays.
_UPLOADER_HAND_ORDER = [
    "wrist", "indexTip", "indexDIP", "indexPIP", "indexMCP",
    "middleTip", "middleDIP", "middlePIP", "middleMCP",
    "ringTip", "ringDIP", "ringPIP", "ringMCP",
    "littleTip", "littleDIP", "littlePIP", "littleMCP",
    "thumbTip", "thumbIP", "thumbMP", "thumbCMC",
]  # fmt: skip
_MEDIAPIPE_HAND_INDEX = {
    "wrist": 0, "thumbCMC": 1, "thumbMP": 2, "thumbIP": 3, "thumbTip": 4,
    "indexMCP": 5, "indexPIP": 6, "indexDIP": 7, "indexTip": 8,
    "middleMCP": 9, "middlePIP": 10, "middleDIP": 11, "middleTip": 12,
    "ringMCP": 13, "ringPIP": 14, "ringDIP": 15, "ringTip": 16,
    "littleMCP": 17, "littlePIP": 18, "littleDIP": 19, "littleTip": 20,
}  # fmt: skip


def test_mediapipe_layout_reads_the_uploaders_interleaved_hand_order() -> None:
    from silent_signal.pose.layouts import MEDIAPIPE_POSE_NAMES, MEDIAPIPE_UPPER68_V1

    raw = np.full((76, 3), -1.0)
    raw[:33, 0] = np.arange(33)  # pose landmarks keep MediaPipe's own indices
    raw[33, 0] = 99  # synthesized neck
    for position, name in enumerate(_UPLOADER_HAND_ORDER):
        for side in (0, 1):
            raw[34 + 2 * position + side] = (100 + side, _MEDIAPIPE_HAND_INDEX[name], 0)
    selected = raw[list(MEDIAPIPE_UPPER68_V1.source_indices)]

    for node, joint in enumerate(MEDIAPIPE_UPPER68_V1.joints):
        if joint.body_part in {"left_hand", "right_hand"}:
            side = 0 if joint.body_part == "left_hand" else 1
            offset = MEDIAPIPE_UPPER68_V1.index_by_name[f"{joint.body_part}_wrist"]
            assert tuple(selected[node, :2]) == (100 + side, node - offset), joint.name
        elif joint.name == "neck":
            assert selected[node, 0] == 99
        else:
            assert MEDIAPIPE_POSE_NAMES[int(selected[node, 0])] == joint.name


def test_mediapipe_hand_edges_match_mediapipe_hand_connections() -> None:
    from silent_signal.pose.layouts import MEDIAPIPE_UPPER68_V1

    connections = {
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
        (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
        (17, 18), (18, 19), (19, 20),
    }  # fmt: skip
    names = MEDIAPIPE_UPPER68_V1.index_by_name
    for hand, body_wrist in (("left_hand", "left_wrist"), ("right_hand", "right_wrist")):
        offset = names[f"{hand}_wrist"]
        local = {
            (a - offset, b - offset)
            for a, b in MEDIAPIPE_UPPER68_V1.edges
            if offset <= a < offset + 21 and offset <= b < offset + 21
        }
        # MediaPipe's connections plus wrist links to the middle and ring MCPs (tree parents).
        assert local == connections | {(0, 9), (0, 13)}
        assert (names[body_wrist], offset) in MEDIAPIPE_UPPER68_V1.edges
