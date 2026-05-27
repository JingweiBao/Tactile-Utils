from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np

from offline_shape_alignment.alignment import apply_similarity_to_points, estimate_similarity_transform
from offline_shape_alignment.types import KeypointSet, ensure_keypoint_set
from offline_shape_alignment.xhand import infer_xhand_semantic_keypoints


DEFAULT_INITIAL_STEP_RAD = 0.35
DEFAULT_MIN_STEP_RAD = 0.02
DEFAULT_ITERATIONS = 5


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float


@dataclass(frozen=True)
class ReferencePoseFit:
    side: str
    qpos: dict[str, float]
    optimized_joints: tuple[str, ...]
    objective_m: float
    baseline_objective_m: float
    iterations: int
    initial_step_rad: float
    min_step_rad: float
    history: tuple[dict[str, float | str], ...]


def fit_xhand_reference_pose(
    urdf_path: str | Path,
    side: str,
    mano_keypoints: KeypointSet,
    *,
    qpos_initial: Mapping[str, float] | None = None,
    joint_names: tuple[str, ...] | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    initial_step_rad: float = DEFAULT_INITIAL_STEP_RAD,
    min_step_rad: float = DEFAULT_MIN_STEP_RAD,
) -> ReferencePoseFit:
    ensure_keypoint_set(mano_keypoints)
    side = _validate_side(side)
    urdf_path = Path(urdf_path)
    joint_names = joint_names or default_xhand_reference_pose_joints(side)
    limits = _load_joint_limits(urdf_path, joint_names)
    qpos = _initial_qpos(qpos_initial or {}, joint_names, limits)

    baseline = _objective(urdf_path, side, qpos, mano_keypoints)
    best_score = baseline
    history: list[dict[str, float | str]] = []
    step = float(initial_step_rad)

    for iteration in range(int(iterations)):
        improved = False
        for joint_name in joint_names:
            current = qpos[joint_name]
            candidates = _candidate_values(current, step, limits[joint_name])
            local_best_value = current
            local_best_score = best_score
            for value in candidates:
                trial = dict(qpos)
                trial[joint_name] = value
                score = _objective(urdf_path, side, trial, mano_keypoints)
                if score < local_best_score:
                    local_best_value = value
                    local_best_score = score

            if local_best_score < best_score:
                qpos[joint_name] = local_best_value
                best_score = local_best_score
                improved = True
                history.append(
                    {
                        "iteration": float(iteration),
                        "joint": joint_name,
                        "value_rad": float(local_best_value),
                        "objective_m": float(best_score),
                    }
                )

        step = max(float(min_step_rad), step * 0.5)
        if not improved and step <= min_step_rad + 1e-12:
            break

    return ReferencePoseFit(
        side=side,
        qpos={name: float(qpos[name]) for name in joint_names},
        optimized_joints=tuple(joint_names),
        objective_m=float(best_score),
        baseline_objective_m=float(baseline),
        iterations=int(iterations),
        initial_step_rad=float(initial_step_rad),
        min_step_rad=float(min_step_rad),
        history=tuple(history),
    )


def default_xhand_reference_pose_joints(side: str) -> tuple[str, ...]:
    side = _validate_side(side)
    return (
        f"{side}_hand_thumb_bend_joint",
        f"{side}_hand_thumb_rota_joint1",
        f"{side}_hand_index_bend_joint",
        f"{side}_hand_index_joint1",
        f"{side}_hand_mid_joint1",
        f"{side}_hand_ring_joint1",
        f"{side}_hand_pinky_joint1",
    )


def reference_pose_fit_to_json(fit: ReferencePoseFit) -> dict[str, object]:
    return {
        "side": fit.side,
        "qpos": fit.qpos,
        "optimized_joints": list(fit.optimized_joints),
        "objective_m": fit.objective_m,
        "baseline_objective_m": fit.baseline_objective_m,
        "improvement_m": fit.baseline_objective_m - fit.objective_m,
        "iterations": fit.iterations,
        "initial_step_rad": fit.initial_step_rad,
        "min_step_rad": fit.min_step_rad,
        "history": [dict(item) for item in fit.history],
    }


def _objective(
    urdf_path: Path,
    side: str,
    qpos: Mapping[str, float],
    mano_keypoints: KeypointSet,
) -> float:
    robot_keypoints = infer_xhand_semantic_keypoints(urdf_path, side, qpos=qpos)
    transform = estimate_similarity_transform(robot_keypoints.points, mano_keypoints.points, allow_reflection=False)
    aligned = apply_similarity_to_points(transform.matrix, robot_keypoints.points)
    residual = aligned - mano_keypoints.points
    weights = _keypoint_weights(mano_keypoints.labels)
    return float(np.sqrt(np.sum(weights * np.sum(residual**2, axis=1)) / np.sum(weights)))


def _keypoint_weights(labels: tuple[str, ...]) -> np.ndarray:
    weights = []
    for label in labels:
        if label == "wrist":
            weights.append(0.5)
        elif label.endswith("_mcp"):
            weights.append(1.5)
        elif label.endswith("_pip"):
            weights.append(1.25)
        elif label.endswith("_tip"):
            weights.append(1.2)
        else:
            weights.append(1.0)
    return np.asarray(weights, dtype=np.float64)


def _load_joint_limits(urdf_path: Path, joint_names: tuple[str, ...]) -> dict[str, JointLimit]:
    root = ET.parse(urdf_path).getroot()
    joint_elements = {joint_el.attrib["name"]: joint_el for joint_el in root.findall("joint")}
    limits: dict[str, JointLimit] = {}
    for name in joint_names:
        joint_el = joint_elements.get(name)
        if joint_el is None:
            raise ValueError(f"XHand URDF {urdf_path} is missing joint {name!r}")
        limit_el = joint_el.find("limit")
        if limit_el is None:
            limits[name] = JointLimit(lower=-np.pi, upper=np.pi)
        else:
            limits[name] = JointLimit(
                lower=float(limit_el.attrib.get("lower", -np.pi)),
                upper=float(limit_el.attrib.get("upper", np.pi)),
            )
    return limits


def _initial_qpos(
    qpos_initial: Mapping[str, float],
    joint_names: tuple[str, ...],
    limits: Mapping[str, JointLimit],
) -> dict[str, float]:
    out = {}
    for name in joint_names:
        value = float(qpos_initial.get(name, 0.0))
        out[name] = _clip(value, limits[name])
    return out


def _candidate_values(current: float, step: float, limit: JointLimit) -> tuple[float, ...]:
    raw = (current - step, current, current + step)
    values = sorted({_clip(value, limit) for value in raw})
    return tuple(values)


def _clip(value: float, limit: JointLimit) -> float:
    return min(max(float(value), limit.lower), limit.upper)


def _validate_side(side: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    return side
