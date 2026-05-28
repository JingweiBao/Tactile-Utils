from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    """Return a rotation matrix for URDF roll-pitch-yaw angles."""
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def transform_matrix(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rpy_matrix(rpy)
    mat[:3, 3] = np.asarray(list(xyz), dtype=np.float64)
    return mat


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return (transform @ np.concatenate([points, ones], axis=1).T).T[:, :3]


def taxel_grid_points(size_mm: Iterable[float], grid_shape: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return local sensor-plane taxel centers and integer row/col indexes."""
    height_mm, width_mm = [float(v) for v in size_mm]
    rows, cols = [int(v) for v in grid_shape]
    if rows <= 0 or cols <= 0:
        raise ValueError(f"grid_shape must be positive, got {grid_shape}")

    ys = np.linspace(-height_mm / 2.0, height_mm / 2.0, rows, dtype=np.float64) / 1000.0
    xs = np.linspace(-width_mm / 2.0, width_mm / 2.0, cols, dtype=np.float64) / 1000.0
    points = []
    indices = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            points.append([x, y, 0.0])
            indices.append([row, col])
    return np.asarray(points, dtype=np.float64), np.asarray(indices, dtype=np.int64)
