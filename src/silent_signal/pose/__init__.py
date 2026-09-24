"""Pose extraction contracts, layouts, and backends."""

from silent_signal.pose.interface import PoseCandidate, RawPoseSequence, WholeBodyExtractor
from silent_signal.pose.layouts import COCO_WHOLEBODY_75_V1, PoseLayout

__all__ = [
    "COCO_WHOLEBODY_75_V1",
    "PoseCandidate",
    "PoseLayout",
    "RawPoseSequence",
    "WholeBodyExtractor",
]
