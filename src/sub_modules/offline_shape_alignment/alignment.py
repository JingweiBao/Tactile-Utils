from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sub_modules.offline_shape_alignment.types import KeypointSet, Mesh, ensure_keypoint_set


MEAN_DISTANCE_WARN_MM = 25.0
MAX_DISTANCE_WARN_MM = 50.0
MIRROR_IMPROVEMENT_RATIO = 0.80
MIN_REASONABLE_SCALE = 0.5
MAX_REASONABLE_SCALE = 2.0


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    matrix: np.ndarray
    rms: float
    determinant: float


def diagnose_alignment(
    robot_keypoints: KeypointSet,
    mano_keypoints: KeypointSet,
    robot_mesh: Mesh | None = None,
    mano_mesh: Mesh | None = None,
) -> dict[str, Any]:
    ensure_keypoint_set(robot_keypoints)
    ensure_keypoint_set(mano_keypoints)
    if robot_keypoints.side != mano_keypoints.side:
        raise ValueError(f"side mismatch: robot={robot_keypoints.side}, mano={mano_keypoints.side}")

    proper = estimate_similarity_transform(robot_keypoints.points, mano_keypoints.points, allow_reflection=False)
    reflected = estimate_similarity_transform(robot_keypoints.points, mano_keypoints.points, allow_reflection=True)
    aligned_robot = apply_similarity_to_points(proper.matrix, robot_keypoints.points)
    distances_m = np.linalg.norm(aligned_robot - mano_keypoints.points, axis=1)
    distances_mm = distances_m * 1000.0

    unit_status = _status_for_scale(proper.scale)
    mirror_status = _status_for_mirror(proper.rms, reflected.rms)
    pose_status = _status_for_pose(distances_mm)
    orientation = _orientation_diagnostics(robot_keypoints, mano_keypoints, proper.rotation)

    keypoints_payload = []
    for idx, label in enumerate(robot_keypoints.labels):
        keypoints_payload.append(
            {
                "label": label,
                "robot_raw": _vector(robot_keypoints.points[idx]),
                "robot_aligned": _vector(aligned_robot[idx]),
                "mano": _vector(mano_keypoints.points[idx]),
                "distance_mm": float(distances_mm[idx]),
            }
        )

    report: dict[str, Any] = {
        "side": robot_keypoints.side,
        "labels": list(robot_keypoints.labels),
        "summary": {
            "mean_distance_mm": float(distances_mm.mean()),
            "max_distance_mm": float(distances_mm.max()),
            "rms_distance_mm": float(np.sqrt(np.mean(distances_mm**2))),
            "scale": float(proper.scale),
            "proper_rms_m": float(proper.rms),
            "reflected_rms_m": float(reflected.rms),
        },
        "status": {
            "unit": unit_status,
            "mirror": mirror_status,
            "pose": pose_status,
            "orientation": orientation["status"],
        },
        "orientation": orientation,
        "transform": {
            "robot_to_mano": _matrix(proper.matrix),
            "scale": float(proper.scale),
            "rotation": _matrix(proper.rotation),
            "translation": _vector(proper.translation),
            "rotation_determinant": float(proper.determinant),
        },
        "alternate_reflection_transform": {
            "scale": float(reflected.scale),
            "rms_m": float(reflected.rms),
            "rotation_determinant": float(reflected.determinant),
        },
        "bounds": {
            "robot_raw": _bounds(robot_mesh.vertices if robot_mesh is not None else robot_keypoints.points),
            "robot_aligned": _bounds(
                apply_similarity_to_points(proper.matrix, robot_mesh.vertices)
                if robot_mesh is not None and robot_mesh.vertices.size
                else aligned_robot
            ),
            "mano": _bounds(mano_mesh.vertices if mano_mesh is not None else mano_keypoints.points),
        },
        "keypoints": keypoints_payload,
        "sources": {
            "robot": _jsonable(robot_keypoints.metadata),
            "mano": _jsonable(mano_keypoints.metadata),
        },
    }
    return report


def estimate_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    allow_reflection: bool,
) -> SimilarityTransform:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"expected matching Nx3 point clouds, got {source.shape} and {target.shape}")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1e-18:
        raise ValueError("source points are degenerate")

    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if not allow_reflection and np.linalg.det(u @ vt) < 0.0:
        signs[-1] = -1.0
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(singular_values * signs) / variance)
    translation = target_mean - scale * (rotation @ source_mean)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    aligned = apply_similarity_to_points(matrix, source)
    rms = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return SimilarityTransform(
        scale=scale,
        rotation=rotation,
        translation=translation,
        matrix=matrix,
        rms=rms,
        determinant=float(np.linalg.det(rotation)),
    )


def apply_similarity_to_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points.reshape((-1, 3)).copy()
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return (np.asarray(matrix, dtype=np.float64) @ np.concatenate([points, ones], axis=1).T).T[:, :3]


def _status_for_scale(scale: float) -> dict[str, object]:
    ok = MIN_REASONABLE_SCALE <= scale <= MAX_REASONABLE_SCALE
    return {
        "level": "ok" if ok else "warn",
        "message": "scale is within expected range" if ok else "scale suggests unit mismatch",
    }


def _status_for_mirror(proper_rms: float, reflected_rms: float) -> dict[str, object]:
    suspect = reflected_rms < proper_rms * MIRROR_IMPROVEMENT_RATIO
    return {
        "level": "warn" if suspect else "ok",
        "message": "reflection fit is much better; check left/right mirror" if suspect else "no strong mirror evidence",
    }


def _status_for_pose(distances_mm: np.ndarray) -> dict[str, object]:
    warn = float(distances_mm.mean()) > MEAN_DISTANCE_WARN_MM or float(distances_mm.max()) > MAX_DISTANCE_WARN_MM
    return {
        "level": "warn" if warn else "ok",
        "message": "reference pose or semantic keypoints may not match" if warn else "keypoint distances are within thresholds",
    }


def _orientation_diagnostics(
    robot_keypoints: KeypointSet,
    mano_keypoints: KeypointSet,
    rotation: np.ndarray,
) -> dict[str, object]:
    robot_basis = _hand_basis(robot_keypoints)
    mano_basis = _hand_basis(mano_keypoints)
    rotated_robot_basis = rotation @ robot_basis
    basis_delta = mano_basis.T @ rotated_robot_basis
    determinant = float(np.linalg.det(basis_delta))
    palm_normal_dot = float(np.dot(rotated_robot_basis[:, 2], mano_basis[:, 2]))
    warn = determinant < 0.0 or palm_normal_dot < 0.5
    return {
        "basis_determinant": determinant,
        "palm_normal_dot": palm_normal_dot,
        "status": {
            "level": "warn" if warn else "ok",
            "message": "palm/back orientation or handedness needs review" if warn else "basis orientation is consistent after alignment",
        },
    }


def _hand_basis(keypoints: KeypointSet) -> np.ndarray:
    by_label = keypoints.point_by_label()
    wrist = by_label["wrist"]
    distal = _unit(by_label["middle_mcp"] - wrist)
    lateral = _unit(by_label["index_mcp"] - by_label["pinky_mcp"])
    normal = _unit(np.cross(lateral, distal))
    if float(np.linalg.norm(normal)) < 1e-9:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    lateral = _unit(np.cross(distal, normal))
    return np.stack([lateral, distal, normal], axis=1)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _bounds(points: np.ndarray) -> dict[str, list[float]]:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "extent": [0.0, 0.0, 0.0]}
    return {
        "min": _vector(points.min(axis=0)),
        "max": _vector(points.max(axis=0)),
        "extent": _vector(points.max(axis=0) - points.min(axis=0)),
    }


def _vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _matrix(values: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(values, dtype=np.float64)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
