from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from online.tactile_mapping import (
    MANO_FULL_POSE_LABELS,
    OnlineReference,
    merge_qpos_with_reference,
    xhand_qpos_to_mano_pose,
)
from sub_modules.offline_shape_alignment.alignment import apply_similarity_to_points
from sub_modules.offline_shape_alignment.types import KEYPOINT_LABELS
from sub_modules.offline_shape_alignment.xhand import infer_xhand_semantic_keypoints
from sub_modules.tactile_layout_3d_projection.urdf import load_urdf


RETARGET_MODEL_VERSION = 1


@dataclass(frozen=True)
class PoseIKConfig:
    iterations: int = 160
    learning_rate: float = 0.03
    keypoint_loss_weight: float = 1.0
    direction_loss_weight: float = 0.15
    pose_prior_weight: float = 1e-3
    root_prior_weight: float = 5e-4
    pose_limit_rad: float = 3.0
    log_every: int = 25


@dataclass(frozen=True)
class PoseIKResult:
    qpos: dict[str, float]
    initial_pose: np.ndarray
    optimized_pose: np.ndarray
    target_keypoints_mano: np.ndarray
    initial_keypoint_rmse_mm: float
    final_keypoint_rmse_mm: float
    final_keypoint_mean_mm: float
    final_keypoint_max_mm: float
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class RetargetRegressionModel:
    side: str
    qpos_names: tuple[str, ...]
    pose_labels: tuple[str, ...]
    coefficients: np.ndarray
    qpos_mean: np.ndarray
    qpos_scale: np.ndarray
    ridge_lambda: float
    feature_kind: str
    metadata: Mapping[str, Any]

    def predict(
        self,
        qpos: Mapping[str, float] | None,
        *,
        reference_qpos: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        qpos_values = qpos_mapping_to_vector(
            qpos or {},
            self.qpos_names,
            reference_qpos=reference_qpos,
        )[None, :]
        pose = predict_retarget_pose(self, qpos_values)[0]
        return pose


@dataclass(frozen=True)
class RetargetCalibrationResult:
    side: str
    qpos_names: tuple[str, ...]
    qpos_values: np.ndarray
    optimized_poses: np.ndarray
    initial_poses: np.ndarray
    initial_keypoint_rmse_mm: np.ndarray
    optimized_keypoint_rmse_mm: np.ndarray
    model: RetargetRegressionModel
    ik_config: PoseIKConfig
    metadata: Mapping[str, Any]


def calibrate_retarget_model(
    reference: OnlineReference,
    qpos_samples: Sequence[Mapping[str, float]],
    *,
    qpos_names: Sequence[str] | None = None,
    ik_config: PoseIKConfig | None = None,
    ridge_lambda: float = 1e-4,
) -> RetargetCalibrationResult:
    if not qpos_samples:
        raise ValueError("qpos_samples must not be empty")
    ik_config = ik_config or PoseIKConfig()
    qpos_names_tuple = tuple(str(name) for name in (qpos_names or infer_xhand_revolute_qpos_names(reference.xhand_urdf_path, reference.side)))
    qpos_values = np.stack(
        [
            qpos_mapping_to_vector(sample, qpos_names_tuple, reference_qpos=reference.reference_qpos)
            for sample in qpos_samples
        ],
        axis=0,
    )

    ik_results: list[PoseIKResult] = []
    for sample in qpos_samples:
        ik_results.append(optimize_mano_pose_for_qpos(reference, sample, config=ik_config))

    initial_poses = np.stack([result.initial_pose for result in ik_results], axis=0)
    optimized_poses = np.stack([result.optimized_pose for result in ik_results], axis=0)
    initial_rmse = np.asarray([result.initial_keypoint_rmse_mm for result in ik_results], dtype=np.float64)
    optimized_rmse = np.asarray([result.final_keypoint_rmse_mm for result in ik_results], dtype=np.float64)
    model = fit_ridge_retarget_model(
        reference.side,
        qpos_names_tuple,
        qpos_values,
        optimized_poses,
        ridge_lambda=ridge_lambda,
        metadata={
            "source": "pose_only_ik_calibration",
            "sample_count": int(len(qpos_samples)),
            "ik_config": asdict(ik_config),
            "initial_keypoint_rmse_mm": _stats(initial_rmse),
            "optimized_keypoint_rmse_mm": _stats(optimized_rmse),
        },
    )
    return RetargetCalibrationResult(
        side=reference.side,
        qpos_names=qpos_names_tuple,
        qpos_values=qpos_values,
        optimized_poses=optimized_poses,
        initial_poses=initial_poses,
        initial_keypoint_rmse_mm=initial_rmse,
        optimized_keypoint_rmse_mm=optimized_rmse,
        model=model,
        ik_config=ik_config,
        metadata={
            "reference": dict(reference.metadata),
            "initial_keypoint_rmse_mm": _stats(initial_rmse),
            "optimized_keypoint_rmse_mm": _stats(optimized_rmse),
        },
    )


def optimize_mano_pose_for_qpos(
    reference: OnlineReference,
    qpos: Mapping[str, float] | None,
    *,
    config: PoseIKConfig | None = None,
    initial_pose: np.ndarray | None = None,
) -> PoseIKResult:
    torch = _require_torch()
    config = config or PoseIKConfig()
    merged_qpos = merge_qpos_with_reference(reference.reference_qpos, qpos)
    target_keypoints = _target_xhand_keypoints_mano(reference, merged_qpos)
    initial = (
        np.asarray(initial_pose, dtype=np.float64)
        if initial_pose is not None
        else xhand_qpos_to_mano_pose(reference, merged_qpos, include_root=True)
    )
    if initial.shape != (len(MANO_FULL_POSE_LABELS), 3):
        raise ValueError(f"initial_pose must have shape ({len(MANO_FULL_POSE_LABELS)}, 3), got {initial.shape}")

    device = reference.model.v_template.device
    dtype = reference.model.v_template.dtype
    beta = torch.as_tensor(reference.beta, dtype=dtype, device=device)
    target = torch.as_tensor(target_keypoints, dtype=dtype, device=device)
    key_weights = torch.as_tensor(_keypoint_weights(), dtype=dtype, device=device)
    bone_pairs = torch.as_tensor(_bone_pairs(), dtype=torch.long, device=device)
    initial_tensor = torch.as_tensor(initial, dtype=dtype, device=device)
    pose = initial_tensor.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pose], lr=float(config.learning_rate))
    history: list[dict[str, float]] = []
    log_every = max(1, int(config.log_every))

    with torch.no_grad():
        initial_losses = _compute_pose_ik_losses(reference.model, beta, initial_tensor, target, key_weights, bone_pairs, initial_tensor, config)
    history.append(_losses_to_history(0, initial_losses))

    for step in range(int(config.iterations)):
        optimizer.zero_grad(set_to_none=True)
        losses = _compute_pose_ik_losses(reference.model, beta, pose, target, key_weights, bone_pairs, initial_tensor, config)
        losses["total"].backward()
        optimizer.step()
        if config.pose_limit_rad > 0.0:
            with torch.no_grad():
                pose.clamp_(min=-float(config.pose_limit_rad), max=float(config.pose_limit_rad))
        if (step + 1) % log_every == 0:
            with torch.no_grad():
                logged = _compute_pose_ik_losses(reference.model, beta, pose, target, key_weights, bone_pairs, initial_tensor, config)
            history.append(_losses_to_history(step + 1, logged))

    with torch.no_grad():
        final_losses = _compute_pose_ik_losses(reference.model, beta, pose, target, key_weights, bone_pairs, initial_tensor, config)
        vertices, joints = reference.model.vertices_joints_from_beta_pose(beta, pose)
        keypoints = reference.model.keypoints_from_vertices(vertices, joints)
        distances_mm = torch.linalg.norm(keypoints - target, dim=1) * 1000.0
    final_history = _losses_to_history(int(config.iterations), final_losses)
    if not history or history[-1]["step"] != final_history["step"]:
        history.append(final_history)

    initial_rmse_mm = float(np.sqrt(max(history[0]["key_mse_m2"], 0.0)) * 1000.0)
    return PoseIKResult(
        qpos={str(name): float(value) for name, value in sorted(merged_qpos.items())},
        initial_pose=initial.copy(),
        optimized_pose=pose.detach().cpu().numpy().astype(np.float64),
        target_keypoints_mano=target_keypoints,
        initial_keypoint_rmse_mm=initial_rmse_mm,
        final_keypoint_rmse_mm=float(torch.sqrt((distances_mm * distances_mm).mean()).detach().cpu()),
        final_keypoint_mean_mm=float(distances_mm.mean().detach().cpu()),
        final_keypoint_max_mm=float(distances_mm.max().detach().cpu()),
        history=tuple(history),
    )


def fit_ridge_retarget_model(
    side: str,
    qpos_names: Sequence[str],
    qpos_values: np.ndarray,
    poses: np.ndarray,
    *,
    ridge_lambda: float = 1e-4,
    metadata: Mapping[str, Any] | None = None,
) -> RetargetRegressionModel:
    qpos = np.asarray(qpos_values, dtype=np.float64)
    pose_values = np.asarray(poses, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"qpos_values must have shape (N, J), got {qpos.shape}")
    if pose_values.ndim != 3 or pose_values.shape[1:] != (len(MANO_FULL_POSE_LABELS), 3):
        raise ValueError(
            f"poses must have shape (N, {len(MANO_FULL_POSE_LABELS)}, 3), got {pose_values.shape}"
        )
    if qpos.shape[0] != pose_values.shape[0]:
        raise ValueError("qpos_values and poses must have the same sample count")
    qpos_names_tuple = tuple(str(name) for name in qpos_names)
    if len(qpos_names_tuple) != qpos.shape[1]:
        raise ValueError(f"qpos_names length {len(qpos_names_tuple)} does not match qpos width {qpos.shape[1]}")
    qpos_mean = qpos.mean(axis=0)
    qpos_scale = qpos.std(axis=0)
    qpos_scale[qpos_scale < 1e-9] = 1.0
    features = _qpos_features(qpos, qpos_mean, qpos_scale)
    target = pose_values.reshape(pose_values.shape[0], -1)
    regularizer = np.eye(features.shape[1], dtype=np.float64) * float(ridge_lambda)
    regularizer[0, 0] = 0.0
    lhs = features.T @ features + regularizer
    rhs = features.T @ target
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs) @ rhs
    return RetargetRegressionModel(
        side=str(side),
        qpos_names=qpos_names_tuple,
        pose_labels=MANO_FULL_POSE_LABELS,
        coefficients=coefficients.astype(np.float64),
        qpos_mean=qpos_mean.astype(np.float64),
        qpos_scale=qpos_scale.astype(np.float64),
        ridge_lambda=float(ridge_lambda),
        feature_kind="bias+zscore+zscore2+sin+cos",
        metadata=dict(metadata or {}),
    )


def predict_retarget_pose(model: RetargetRegressionModel, qpos_values: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos_values, dtype=np.float64)
    single = qpos.ndim == 1
    if single:
        qpos = qpos[None, :]
    if qpos.ndim != 2 or qpos.shape[1] != len(model.qpos_names):
        raise ValueError(f"qpos_values must have shape (N, {len(model.qpos_names)}), got {qpos.shape}")
    features = _qpos_features(qpos, model.qpos_mean, model.qpos_scale)
    flat_pose = features @ np.asarray(model.coefficients, dtype=np.float64)
    poses = flat_pose.reshape(qpos.shape[0], len(MANO_FULL_POSE_LABELS), 3)
    return poses[0] if single else poses


def save_retarget_model(
    model: RetargetRegressionModel,
    out_path: str | Path,
    *,
    calibration: RetargetCalibrationResult | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": np.asarray(RETARGET_MODEL_VERSION, dtype=np.int64),
        "side": np.asarray(model.side),
        "qpos_names": np.asarray(model.qpos_names),
        "pose_labels": np.asarray(model.pose_labels),
        "coefficients": model.coefficients.astype(np.float64),
        "qpos_mean": model.qpos_mean.astype(np.float64),
        "qpos_scale": model.qpos_scale.astype(np.float64),
        "ridge_lambda": np.asarray(model.ridge_lambda, dtype=np.float64),
        "feature_kind": np.asarray(model.feature_kind),
        "metadata_json": np.asarray(json.dumps(_jsonable(model.metadata))),
    }
    if calibration is not None:
        payload.update(
            {
                "qpos_values": calibration.qpos_values.astype(np.float64),
                "optimized_poses": calibration.optimized_poses.astype(np.float64),
                "initial_poses": calibration.initial_poses.astype(np.float64),
                "initial_keypoint_rmse_mm": calibration.initial_keypoint_rmse_mm.astype(np.float64),
                "optimized_keypoint_rmse_mm": calibration.optimized_keypoint_rmse_mm.astype(np.float64),
                "calibration_metadata_json": np.asarray(json.dumps(_jsonable(calibration.metadata))),
            }
        )
    np.savez_compressed(out_path, **payload)


def load_retarget_model(path: str | Path) -> RetargetRegressionModel:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        version = int(np.asarray(payload["version"]).item()) if "version" in payload.files else 0
        if version != RETARGET_MODEL_VERSION:
            raise ValueError(f"unsupported retarget model version {version}; expected {RETARGET_MODEL_VERSION}")
        metadata_raw = str(np.asarray(payload["metadata_json"]).item()) if "metadata_json" in payload.files else "{}"
        return RetargetRegressionModel(
            side=str(np.asarray(payload["side"]).item()),
            qpos_names=tuple(str(item) for item in np.asarray(payload["qpos_names"]).tolist()),
            pose_labels=tuple(str(item) for item in np.asarray(payload["pose_labels"]).tolist()),
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
            qpos_mean=np.asarray(payload["qpos_mean"], dtype=np.float64),
            qpos_scale=np.asarray(payload["qpos_scale"], dtype=np.float64),
            ridge_lambda=float(np.asarray(payload["ridge_lambda"]).item()),
            feature_kind=str(np.asarray(payload["feature_kind"]).item()),
            metadata=json.loads(metadata_raw),
        )


def sample_qpos_uniform(
    qpos_names: Sequence[str],
    qpos_min: Mapping[str, float],
    qpos_max: Mapping[str, float],
    *,
    count: int,
    seed: int = 7,
) -> list[dict[str, float]]:
    if count <= 0:
        raise ValueError("count must be positive")
    names = tuple(str(name) for name in qpos_names)
    lows = np.asarray([float(qpos_min[name]) for name in names], dtype=np.float64)
    highs = np.asarray([float(qpos_max[name]) for name in names], dtype=np.float64)
    if np.any(highs < lows):
        raise ValueError("qpos_max must be greater than or equal to qpos_min for every joint")
    rng = np.random.default_rng(int(seed))
    values = rng.uniform(lows[None, :], highs[None, :], size=(int(count), len(names)))
    return [vector_to_qpos_mapping(row, names) for row in values]


def default_qpos_ranges(
    qpos_names: Sequence[str],
    *,
    qpos_min: float = 0.0,
    qpos_max: float = 1.35,
    reference_qpos: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    mins = {str(name): float(qpos_min) for name in qpos_names}
    maxs = {str(name): float(qpos_max) for name in qpos_names}
    for name, value in dict(reference_qpos or {}).items():
        if name in mins:
            mins[name] = min(mins[name], float(value))
            maxs[name] = max(maxs[name], float(value))
    return mins, maxs


def infer_xhand_revolute_qpos_names(urdf_path: str | Path, side: str) -> tuple[str, ...]:
    model = load_urdf(urdf_path)
    prefix = f"{side}_hand_"
    return tuple(joint.name for joint in model.joints if joint.name.startswith(prefix) and joint.joint_type == "revolute")


def qpos_mapping_to_vector(
    qpos: Mapping[str, float],
    qpos_names: Sequence[str],
    *,
    reference_qpos: Mapping[str, float] | None = None,
) -> np.ndarray:
    reference = {str(name): float(value) for name, value in dict(reference_qpos or {}).items()}
    values = []
    for name in qpos_names:
        key = str(name)
        values.append(float(qpos[key]) if key in qpos else reference.get(key, 0.0))
    return np.asarray(values, dtype=np.float64)


def vector_to_qpos_mapping(values: np.ndarray, qpos_names: Sequence[str]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    names = tuple(str(name) for name in qpos_names)
    if values.shape != (len(names),):
        raise ValueError(f"values must have shape ({len(names)},), got {values.shape}")
    return {name: float(values[idx]) for idx, name in enumerate(names)}


def retarget_calibration_to_json(
    calibration: RetargetCalibrationResult,
    *,
    outputs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    optimized = np.asarray(calibration.optimized_keypoint_rmse_mm, dtype=np.float64)
    initial = np.asarray(calibration.initial_keypoint_rmse_mm, dtype=np.float64)
    pose_pred = predict_retarget_pose(calibration.model, calibration.qpos_values)
    pose_rmse = np.sqrt(np.mean((pose_pred - calibration.optimized_poses) ** 2, axis=(1, 2)))
    return {
        "module": "online_retarget_calibration",
        "side": calibration.side,
        "sample_count": int(calibration.qpos_values.shape[0]),
        "qpos_names": list(calibration.qpos_names),
        "pose_labels": list(calibration.model.pose_labels),
        "ik_config": asdict(calibration.ik_config),
        "ik_initial_keypoint_rmse_mm": _stats(initial),
        "ik_optimized_keypoint_rmse_mm": _stats(optimized),
        "regression_pose_rmse_rad": _stats(pose_rmse),
        "model": {
            "feature_kind": calibration.model.feature_kind,
            "ridge_lambda": float(calibration.model.ridge_lambda),
            "coefficient_shape": list(calibration.model.coefficients.shape),
        },
        "outputs": dict(outputs or {}),
        "metadata": _jsonable(calibration.metadata),
    }


def _target_xhand_keypoints_mano(reference: OnlineReference, qpos: Mapping[str, float]) -> np.ndarray:
    keypoints = infer_xhand_semantic_keypoints(reference.xhand_urdf_path, reference.side, qpos=qpos)
    return apply_similarity_to_points(reference.robot_to_mano, keypoints.points).astype(np.float64)


def _compute_pose_ik_losses(model, beta, pose, target, key_weights, bone_pairs, initial_pose, config: PoseIKConfig):
    torch = _require_torch()
    vertices, joints = model.vertices_joints_from_beta_pose(beta, pose)
    keypoints = model.keypoints_from_vertices(vertices, joints)
    residual = keypoints - target
    key_mse = (key_weights * (residual * residual).sum(dim=1)).sum() / key_weights.sum()
    src_dirs = keypoints[bone_pairs[:, 1]] - keypoints[bone_pairs[:, 0]]
    dst_dirs = target[bone_pairs[:, 1]] - target[bone_pairs[:, 0]]
    src_dirs = src_dirs / torch.linalg.norm(src_dirs, dim=1, keepdim=True).clamp_min(1e-8)
    dst_dirs = dst_dirs / torch.linalg.norm(dst_dirs, dim=1, keepdim=True).clamp_min(1e-8)
    direction = ((src_dirs - dst_dirs) * (src_dirs - dst_dirs)).sum(dim=1).mean()
    pose_delta = pose - initial_pose
    pose_prior = (pose_delta[1:] * pose_delta[1:]).mean()
    root_prior = (pose_delta[:1] * pose_delta[:1]).mean()
    total = (
        float(config.keypoint_loss_weight) * key_mse
        + float(config.direction_loss_weight) * direction
        + float(config.pose_prior_weight) * pose_prior
        + float(config.root_prior_weight) * root_prior
    )
    return {
        "total": total,
        "key_mse": key_mse,
        "direction": direction,
        "pose_prior": pose_prior,
        "root_prior": root_prior,
    }


def _losses_to_history(step: int, losses: Mapping[str, Any]) -> dict[str, float]:
    return {
        "step": float(step),
        "total": _torch_item(losses["total"]),
        "key_mse_m2": _torch_item(losses["key_mse"]),
        "key_rmse_mm": float(np.sqrt(max(_torch_item(losses["key_mse"]), 0.0)) * 1000.0),
        "direction": _torch_item(losses["direction"]),
        "pose_prior": _torch_item(losses["pose_prior"]),
        "root_prior": _torch_item(losses["root_prior"]),
    }


def _qpos_features(qpos: np.ndarray, qpos_mean: np.ndarray, qpos_scale: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    z = (qpos - qpos_mean[None, :]) / qpos_scale[None, :]
    return np.concatenate(
        [
            np.ones((qpos.shape[0], 1), dtype=np.float64),
            z,
            z * z,
            np.sin(qpos),
            np.cos(qpos),
        ],
        axis=1,
    )


def _keypoint_weights() -> np.ndarray:
    weights = []
    for label in KEYPOINT_LABELS:
        if label == "wrist":
            weights.append(0.75)
        elif label.endswith("_mcp"):
            weights.append(1.5)
        elif label.endswith("_pip"):
            weights.append(1.25)
        elif label.endswith("_dip"):
            weights.append(1.15)
        elif label.endswith("_tip"):
            weights.append(1.6)
        else:
            weights.append(1.0)
    return np.asarray(weights, dtype=np.float64)


def _bone_pairs() -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(KEYPOINT_LABELS)}
    pairs: list[tuple[int, int]] = []
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        labels = ["wrist", f"{finger}_mcp", f"{finger}_pip", f"{finger}_dip", f"{finger}_tip"]
        pairs.extend((label_to_idx[a], label_to_idx[b]) for a, b in zip(labels[:-1], labels[1:]))
    return np.asarray(pairs, dtype=np.int64)


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def _torch_item(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when optional dependency is unavailable.
        raise RuntimeError(
            "PyTorch is required for retarget calibration pose IK. "
            "Use an environment that can import torch before running calibrate-retarget."
        ) from exc
    return torch
