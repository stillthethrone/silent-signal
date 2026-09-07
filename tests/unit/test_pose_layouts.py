from __future__ import annotations

import numpy as np

from silent_signal.pose.layouts import ASL_CITIZEN_WHOLEBODY_V1


def test_asl_layout_is_a_stable_75_joint_coco_wholebody_subset() -> None:
    layout = ASL_CITIZEN_WHOLEBODY_V1

    assert layout.name == "asl_citizen_coco_wholebody_v1"
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
    layout = ASL_CITIZEN_WHOLEBODY_V1
    xy = np.arange(2 * 133 * 2, dtype=np.float32).reshape(2, 133, 2)
    scores = np.arange(2 * 133, dtype=np.float32).reshape(2, 133)

    selected_xy, selected_scores = layout.select(xy, scores)

    assert selected_xy.shape == (2, 75, 2)
    assert selected_scores.shape == (2, 75)
    np.testing.assert_array_equal(selected_xy[:, 13], xy[:, 91])
    np.testing.assert_array_equal(selected_scores[:, 34], scores[:, 112])
