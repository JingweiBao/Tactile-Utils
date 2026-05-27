from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


KEYPOINT_LABELS: tuple[str, ...] = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class KeypointSet:
    side: str
    labels: tuple[str, ...]
    points: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def point_by_label(self) -> dict[str, np.ndarray]:
        return {label: self.points[idx] for idx, label in enumerate(self.labels)}


def ensure_keypoint_set(keypoints: KeypointSet) -> None:
    if keypoints.labels != KEYPOINT_LABELS:
        raise ValueError(f"expected canonical 21 keypoint labels, got {keypoints.labels!r}")
    if keypoints.points.shape != (len(KEYPOINT_LABELS), 3):
        raise ValueError(f"expected keypoints with shape (21, 3), got {keypoints.points.shape}")
