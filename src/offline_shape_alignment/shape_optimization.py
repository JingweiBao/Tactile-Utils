from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from offline_shape_alignment.alignment import apply_similarity_to_points, diagnose_alignment
from offline_shape_alignment.mano import load_mano_reference
from offline_shape_alignment.mano_torch import MANOShapeInstance, load_mano_beta_model
from offline_shape_alignment.reference_pose import fit_xhand_reference_pose, reference_pose_fit_to_json
from offline_shape_alignment.sampling import make_surface_sample_pattern, sample_mesh_surface
from offline_shape_alignment.types import KEYPOINT_LABELS, KeypointSet, Mesh, ensure_keypoint_set
from offline_shape_alignment.xhand import default_xhand_urdf_path, load_xhand_reference


DEFAULT_NUM_BETAS = 10
DEFAULT_ITERATIONS = 1000
DEFAULT_SURFACE_POINTS = 2048
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_KEY_DECAY_STEPS = 2500
DEFAULT_BETA_L2_WEIGHT = 1e-4
DEFAULT_LOG_EVERY = 25
DEFAULT_POSE_ITERATIONS = 600
DEFAULT_BETA_INIT_ITERATIONS = 400
DEFAULT_POSE_LEARNING_RATE = 0.01
DEFAULT_POSE_LIMIT_RAD = 0.25
DEFAULT_POSE_L2_WEIGHT = 1e-3
MANO_POSE_RESIDUAL_LABELS: tuple[str, ...] = (
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
)


@dataclass(frozen=True)
class ShapeOptimizationConfig:
    num_betas: int = DEFAULT_NUM_BETAS
    iterations: int = DEFAULT_ITERATIONS
    robot_surface_points: int = DEFAULT_SURFACE_POINTS
    mano_surface_points: int = DEFAULT_SURFACE_POINTS
    learning_rate: float = DEFAULT_LEARNING_RATE
    key_decay_steps: int = DEFAULT_KEY_DECAY_STEPS
    key_loss_weight: float = 1.0
    beta_l2_weight: float = DEFAULT_BETA_L2_WEIGHT
    log_every: int = DEFAULT_LOG_EVERY
    seed: int = 7
    device: str = "cpu"


@dataclass(frozen=True)
class PoseShapeOptimizationConfig(ShapeOptimizationConfig):
    iterations: int = DEFAULT_POSE_ITERATIONS
    beta_init_iterations: int = DEFAULT_BETA_INIT_ITERATIONS
    pose_learning_rate: float = DEFAULT_POSE_LEARNING_RATE
    pose_limit_rad: float = DEFAULT_POSE_LIMIT_RAD
    pose_l2_weight: float = DEFAULT_POSE_L2_WEIGHT


@dataclass(frozen=True)
class ShapeOptimizationResult:
    side: str
    beta: np.ndarray
    fitted_mano: MANOShapeInstance
    robot_aligned_mesh: Mesh
    robot_aligned_keypoints: KeypointSet
    fixed_frame_report: dict[str, Any]
    initial_alignment_report: dict[str, Any]
    config: ShapeOptimizationConfig
    history: tuple[dict[str, float], ...]
    reference_pose_fit: dict[str, Any] | None
    robot_qpos: dict[str, float]


@dataclass(frozen=True)
class PoseShapeOptimizationResult:
    side: str
    beta: np.ndarray
    pose: np.ndarray
    fitted_mano: MANOShapeInstance
    robot_aligned_mesh: Mesh
    robot_aligned_keypoints: KeypointSet
    fixed_frame_report: dict[str, Any]
    initial_alignment_report: dict[str, Any]
    config: PoseShapeOptimizationConfig
    history: tuple[dict[str, float], ...]
    beta_only_result: ShapeOptimizationResult
    reference_pose_fit: dict[str, Any] | None
    robot_qpos: dict[str, float]


def fit_mano_beta_to_xhand(
    side: str,
    *,
    xhand_root: str | Path,
    mano_root: str | Path,
    config: ShapeOptimizationConfig | None = None,
    fit_reference_pose: bool = True,
    reference_pose_iterations: int = 5,
) -> ShapeOptimizationResult:
    config = config or ShapeOptimizationConfig()
    mano_reference = load_mano_reference(side, mano_root)
    urdf_path = default_xhand_urdf_path(xhand_root, side)

    reference_pose_fit = None
    qpos: dict[str, float] = {}
    if fit_reference_pose:
        fit = fit_xhand_reference_pose(
            urdf_path,
            side,
            mano_reference.keypoints,
            iterations=reference_pose_iterations,
        )
        qpos = dict(fit.qpos)
        reference_pose_fit = reference_pose_fit_to_json(fit)

    robot = load_xhand_reference(urdf_path, side=side, qpos=qpos)
    initial_alignment = diagnose_alignment(robot.keypoints, mano_reference.keypoints, robot.mesh, mano_reference.mesh)
    robot_to_mano = np.asarray(initial_alignment["transform"]["robot_to_mano"], dtype=np.float64)
    robot_aligned_mesh = Mesh(
        vertices=apply_similarity_to_points(robot_to_mano, robot.mesh.vertices),
        faces=robot.mesh.faces.copy(),
    )
    robot_aligned_keypoints = KeypointSet(
        side=side,
        labels=KEYPOINT_LABELS,
        points=apply_similarity_to_points(robot_to_mano, robot.keypoints.points),
        metadata={
            "hand": "xhand",
            "source": "xhand_reference_pose_after_robot_to_mano",
            "qpos": {name: float(value) for name, value in sorted(qpos.items())},
            "robot_to_mano": initial_alignment["transform"]["robot_to_mano"],
        },
    )

    result = fit_mano_beta_to_aligned_robot(
        side,
        robot_aligned_mesh=robot_aligned_mesh,
        robot_aligned_keypoints=robot_aligned_keypoints,
        mano_root=mano_root,
        config=config,
    )
    return ShapeOptimizationResult(
        side=side,
        beta=result.beta,
        fitted_mano=result.fitted_mano,
        robot_aligned_mesh=result.robot_aligned_mesh,
        robot_aligned_keypoints=result.robot_aligned_keypoints,
        fixed_frame_report=result.fixed_frame_report,
        initial_alignment_report=initial_alignment,
        config=result.config,
        history=result.history,
        reference_pose_fit=reference_pose_fit,
        robot_qpos=qpos,
    )


def fit_mano_beta_pose_to_xhand(
    side: str,
    *,
    xhand_root: str | Path,
    mano_root: str | Path,
    config: PoseShapeOptimizationConfig | None = None,
    fit_reference_pose: bool = True,
    reference_pose_iterations: int = 5,
) -> PoseShapeOptimizationResult:
    config = config or PoseShapeOptimizationConfig()
    mano_reference = load_mano_reference(side, mano_root)
    urdf_path = default_xhand_urdf_path(xhand_root, side)

    reference_pose_fit = None
    qpos: dict[str, float] = {}
    if fit_reference_pose:
        fit = fit_xhand_reference_pose(
            urdf_path,
            side,
            mano_reference.keypoints,
            iterations=reference_pose_iterations,
        )
        qpos = dict(fit.qpos)
        reference_pose_fit = reference_pose_fit_to_json(fit)

    robot = load_xhand_reference(urdf_path, side=side, qpos=qpos)
    initial_alignment = diagnose_alignment(robot.keypoints, mano_reference.keypoints, robot.mesh, mano_reference.mesh)
    robot_to_mano = np.asarray(initial_alignment["transform"]["robot_to_mano"], dtype=np.float64)
    robot_aligned_mesh = Mesh(
        vertices=apply_similarity_to_points(robot_to_mano, robot.mesh.vertices),
        faces=robot.mesh.faces.copy(),
    )
    robot_aligned_keypoints = KeypointSet(
        side=side,
        labels=KEYPOINT_LABELS,
        points=apply_similarity_to_points(robot_to_mano, robot.keypoints.points),
        metadata={
            "hand": "xhand",
            "source": "xhand_reference_pose_after_robot_to_mano",
            "qpos": {name: float(value) for name, value in sorted(qpos.items())},
            "robot_to_mano": initial_alignment["transform"]["robot_to_mano"],
        },
    )

    beta_config = ShapeOptimizationConfig(
        num_betas=config.num_betas,
        iterations=config.beta_init_iterations,
        robot_surface_points=config.robot_surface_points,
        mano_surface_points=config.mano_surface_points,
        learning_rate=config.learning_rate,
        key_decay_steps=config.key_decay_steps,
        key_loss_weight=config.key_loss_weight,
        beta_l2_weight=config.beta_l2_weight,
        log_every=config.log_every,
        seed=config.seed,
        device=config.device,
    )
    beta_only = fit_mano_beta_to_aligned_robot(
        side,
        robot_aligned_mesh=robot_aligned_mesh,
        robot_aligned_keypoints=robot_aligned_keypoints,
        mano_root=mano_root,
        config=beta_config,
    )
    result = fit_mano_beta_pose_to_aligned_robot(
        side,
        robot_aligned_mesh=robot_aligned_mesh,
        robot_aligned_keypoints=robot_aligned_keypoints,
        mano_root=mano_root,
        config=config,
        initial_beta=beta_only.beta,
        beta_only_result=beta_only,
    )
    return PoseShapeOptimizationResult(
        side=side,
        beta=result.beta,
        pose=result.pose,
        fitted_mano=result.fitted_mano,
        robot_aligned_mesh=result.robot_aligned_mesh,
        robot_aligned_keypoints=result.robot_aligned_keypoints,
        fixed_frame_report=result.fixed_frame_report,
        initial_alignment_report=initial_alignment,
        config=result.config,
        history=result.history,
        beta_only_result=beta_only,
        reference_pose_fit=reference_pose_fit,
        robot_qpos=qpos,
    )


def fit_mano_beta_to_aligned_robot(
    side: str,
    *,
    robot_aligned_mesh: Mesh,
    robot_aligned_keypoints: KeypointSet,
    mano_root: str | Path,
    config: ShapeOptimizationConfig | None = None,
) -> ShapeOptimizationResult:
    torch = _require_torch()
    config = config or ShapeOptimizationConfig()
    ensure_keypoint_set(robot_aligned_keypoints)

    model = load_mano_beta_model(
        side,
        mano_root,
        num_betas=config.num_betas,
        device=config.device,
        dtype="float32",
    )
    target_surface_np = sample_mesh_surface(
        robot_aligned_mesh,
        config.robot_surface_points,
        seed=config.seed,
    )
    mano_pattern = make_surface_sample_pattern(
        model.instance().mesh.vertices,
        model.faces,
        config.mano_surface_points,
        seed=config.seed + 1,
    )

    device = model.v_template.device
    faces = torch.as_tensor(model.faces, dtype=torch.long, device=device)
    face_indices = torch.as_tensor(mano_pattern.face_indices, dtype=torch.long, device=device)
    barycentric = torch.as_tensor(mano_pattern.barycentric, dtype=model.v_template.dtype, device=device)
    target_surface = torch.as_tensor(target_surface_np, dtype=model.v_template.dtype, device=device)
    target_keypoints = torch.as_tensor(robot_aligned_keypoints.points, dtype=model.v_template.dtype, device=device)
    keypoint_weights = torch.as_tensor(_keypoint_weights(robot_aligned_keypoints.labels), dtype=model.v_template.dtype, device=device)

    beta = torch.zeros(model.num_betas, dtype=model.v_template.dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=float(config.learning_rate))
    history: list[dict[str, float]] = []
    log_every = max(1, int(config.log_every))

    with torch.no_grad():
        initial_losses = _compute_losses(
            model,
            beta,
            faces,
            face_indices,
            barycentric,
            target_surface,
            target_keypoints,
            keypoint_weights,
            step=0,
            config=config,
        )
    history.append(_losses_to_history(0, initial_losses))

    for step in range(int(config.iterations)):
        optimizer.zero_grad(set_to_none=True)
        losses = _compute_losses(
            model,
            beta,
            faces,
            face_indices,
            barycentric,
            target_surface,
            target_keypoints,
            keypoint_weights,
            step=step,
            config=config,
        )
        losses["total"].backward()
        optimizer.step()
        if (step + 1) % log_every == 0:
            with torch.no_grad():
                logged_losses = _compute_losses(
                    model,
                    beta,
                    faces,
                    face_indices,
                    barycentric,
                    target_surface,
                    target_keypoints,
                    keypoint_weights,
                    step=step + 1,
                    config=config,
                )
            history.append(_losses_to_history(step + 1, logged_losses))

    with torch.no_grad():
        final_losses = _compute_losses(
            model,
            beta,
            faces,
            face_indices,
            barycentric,
            target_surface,
            target_keypoints,
            keypoint_weights,
            step=int(config.iterations),
            config=config,
        )
    final_history = _losses_to_history(int(config.iterations), final_losses)
    if not history or history[-1]["step"] != final_history["step"]:
        history.append(final_history)

    beta_np = beta.detach().cpu().numpy().astype(np.float64)
    fitted_mano = model.instance(beta_np)
    fixed_report = build_fixed_frame_report(
        robot_aligned_keypoints,
        fitted_mano.keypoints,
        robot_aligned_mesh,
        fitted_mano.mesh,
    )
    return ShapeOptimizationResult(
        side=side,
        beta=beta_np,
        fitted_mano=fitted_mano,
        robot_aligned_mesh=robot_aligned_mesh,
        robot_aligned_keypoints=robot_aligned_keypoints,
        fixed_frame_report=fixed_report,
        initial_alignment_report={},
        config=config,
        history=tuple(history),
        reference_pose_fit=None,
        robot_qpos={},
    )


def fit_mano_beta_pose_to_aligned_robot(
    side: str,
    *,
    robot_aligned_mesh: Mesh,
    robot_aligned_keypoints: KeypointSet,
    mano_root: str | Path,
    config: PoseShapeOptimizationConfig | None = None,
    initial_beta: np.ndarray | None = None,
    beta_only_result: ShapeOptimizationResult | None = None,
) -> PoseShapeOptimizationResult:
    torch = _require_torch()
    config = config or PoseShapeOptimizationConfig()
    ensure_keypoint_set(robot_aligned_keypoints)

    model = load_mano_beta_model(
        side,
        mano_root,
        num_betas=config.num_betas,
        device=config.device,
        dtype="float32",
    )
    if initial_beta is None:
        initial_beta = np.zeros(config.num_betas, dtype=np.float64)
    target_surface_np = sample_mesh_surface(
        robot_aligned_mesh,
        config.robot_surface_points,
        seed=config.seed,
    )
    mano_pattern = make_surface_sample_pattern(
        model.instance(initial_beta).mesh.vertices,
        model.faces,
        config.mano_surface_points,
        seed=config.seed + 1,
    )

    device = model.v_template.device
    faces = torch.as_tensor(model.faces, dtype=torch.long, device=device)
    face_indices = torch.as_tensor(mano_pattern.face_indices, dtype=torch.long, device=device)
    barycentric = torch.as_tensor(mano_pattern.barycentric, dtype=model.v_template.dtype, device=device)
    target_surface = torch.as_tensor(target_surface_np, dtype=model.v_template.dtype, device=device)
    target_keypoints = torch.as_tensor(robot_aligned_keypoints.points, dtype=model.v_template.dtype, device=device)
    keypoint_weights = torch.as_tensor(_keypoint_weights(robot_aligned_keypoints.labels), dtype=model.v_template.dtype, device=device)

    beta = torch.as_tensor(initial_beta, dtype=model.v_template.dtype, device=device).clone().detach().requires_grad_(True)
    pose_raw = torch.zeros((15, 3), dtype=model.v_template.dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        [
            {"params": [beta], "lr": float(config.learning_rate)},
            {"params": [pose_raw], "lr": float(config.pose_learning_rate)},
        ]
    )
    history: list[dict[str, float]] = []
    log_every = max(1, int(config.log_every))

    with torch.no_grad():
        initial_losses = _compute_pose_losses(
            model,
            beta,
            pose_raw,
            faces,
            face_indices,
            barycentric,
            target_surface,
            target_keypoints,
            keypoint_weights,
            step=0,
            config=config,
        )
    history.append(_losses_to_history(0, initial_losses))

    for step in range(int(config.iterations)):
        optimizer.zero_grad(set_to_none=True)
        losses = _compute_pose_losses(
            model,
            beta,
            pose_raw,
            faces,
            face_indices,
            barycentric,
            target_surface,
            target_keypoints,
            keypoint_weights,
            step=step,
            config=config,
        )
        losses["total"].backward()
        optimizer.step()
        if (step + 1) % log_every == 0:
            with torch.no_grad():
                logged_losses = _compute_pose_losses(
                    model,
                    beta,
                    pose_raw,
                    faces,
                    face_indices,
                    barycentric,
                    target_surface,
                    target_keypoints,
                    keypoint_weights,
                    step=step + 1,
                    config=config,
                )
            history.append(_losses_to_history(step + 1, logged_losses))

    with torch.no_grad():
        final_losses = _compute_pose_losses(
            model,
            beta,
            pose_raw,
            faces,
            face_indices,
            barycentric,
            target_surface,
            target_keypoints,
            keypoint_weights,
            step=int(config.iterations),
            config=config,
        )
        pose = _bounded_pose(pose_raw, config.pose_limit_rad)
    final_history = _losses_to_history(int(config.iterations), final_losses)
    if not history or history[-1]["step"] != final_history["step"]:
        history.append(final_history)

    beta_np = beta.detach().cpu().numpy().astype(np.float64)
    pose_np = pose.detach().cpu().numpy().astype(np.float64)
    fitted_mano = model.instance(beta_np, pose_np)
    fixed_report = build_fixed_frame_report(
        robot_aligned_keypoints,
        fitted_mano.keypoints,
        robot_aligned_mesh,
        fitted_mano.mesh,
    )
    if beta_only_result is None:
        beta_only_result = ShapeOptimizationResult(
            side=side,
            beta=np.asarray(initial_beta, dtype=np.float64),
            fitted_mano=model.instance(initial_beta),
            robot_aligned_mesh=robot_aligned_mesh,
            robot_aligned_keypoints=robot_aligned_keypoints,
            fixed_frame_report=build_fixed_frame_report(
                robot_aligned_keypoints,
                model.instance(initial_beta).keypoints,
                robot_aligned_mesh,
                model.instance(initial_beta).mesh,
            ),
            initial_alignment_report={},
            config=ShapeOptimizationConfig(num_betas=config.num_betas),
            history=tuple(),
            reference_pose_fit=None,
            robot_qpos={},
        )
    return PoseShapeOptimizationResult(
        side=side,
        beta=beta_np,
        pose=pose_np,
        fitted_mano=fitted_mano,
        robot_aligned_mesh=robot_aligned_mesh,
        robot_aligned_keypoints=robot_aligned_keypoints,
        fixed_frame_report=fixed_report,
        initial_alignment_report={},
        config=config,
        history=tuple(history),
        beta_only_result=beta_only_result,
        reference_pose_fit=None,
        robot_qpos={},
    )


def build_fixed_frame_report(
    robot_keypoints: KeypointSet,
    mano_keypoints: KeypointSet,
    robot_mesh: Mesh,
    mano_mesh: Mesh,
) -> dict[str, Any]:
    ensure_keypoint_set(robot_keypoints)
    ensure_keypoint_set(mano_keypoints)
    if robot_keypoints.side != mano_keypoints.side:
        raise ValueError(f"side mismatch: robot={robot_keypoints.side}, mano={mano_keypoints.side}")

    distances_m = np.linalg.norm(robot_keypoints.points - mano_keypoints.points, axis=1)
    distances_mm = distances_m * 1000.0
    keypoints_payload = [
        {
            "label": label,
            "robot_raw": _vector(robot_keypoints.points[idx]),
            "robot_aligned": _vector(robot_keypoints.points[idx]),
            "mano": _vector(mano_keypoints.points[idx]),
            "distance_mm": float(distances_mm[idx]),
        }
        for idx, label in enumerate(robot_keypoints.labels)
    ]
    identity = np.eye(4, dtype=np.float64)
    mean_mm = float(distances_mm.mean())
    max_mm = float(distances_mm.max())
    return {
        "side": robot_keypoints.side,
        "labels": list(robot_keypoints.labels),
        "summary": {
            "mean_distance_mm": mean_mm,
            "max_distance_mm": max_mm,
            "rms_distance_mm": float(np.sqrt(np.mean(distances_mm**2))),
            "scale": 1.0,
            "proper_rms_m": float(np.sqrt(np.mean(distances_m**2))),
            "reflected_rms_m": float(np.sqrt(np.mean(distances_m**2))),
        },
        "status": {
            "unit": {"level": "ok", "message": "fixed MANO frame"},
            "mirror": {"level": "ok", "message": "fixed MANO frame"},
            "pose": {
                "level": "warn" if mean_mm > 25.0 or max_mm > 50.0 else "ok",
                "message": "fixed-frame keypoint distances after beta fitting",
            },
            "orientation": {"level": "ok", "message": "fixed MANO frame"},
        },
        "transform": {
            "robot_to_mano": identity.tolist(),
            "scale": 1.0,
            "rotation": identity[:3, :3].tolist(),
            "translation": [0.0, 0.0, 0.0],
            "rotation_determinant": 1.0,
        },
        "bounds": {
            "robot_raw": _bounds(robot_mesh.vertices),
            "robot_aligned": _bounds(robot_mesh.vertices),
            "mano": _bounds(mano_mesh.vertices),
        },
        "keypoints": keypoints_payload,
        "sources": {
            "robot": _jsonable(robot_keypoints.metadata),
            "mano": _jsonable(mano_keypoints.metadata),
        },
    }


def shape_optimization_result_to_json(result: ShapeOptimizationResult) -> dict[str, Any]:
    history = [dict(item) for item in result.history]
    return {
        "side": result.side,
        "beta": [float(value) for value in result.beta],
        "robot_qpos": {name: float(value) for name, value in sorted(result.robot_qpos.items())},
        "reference_pose_fit": result.reference_pose_fit,
        "config": asdict(result.config),
        "initial_alignment_summary": result.initial_alignment_report.get("summary", {}),
        "fixed_frame_summary": result.fixed_frame_report["summary"],
        "final_losses": history[-1] if history else {},
        "history": history,
        "fingertip_vertex_indices": dict(result.fitted_mano.fingertip_vertex_indices),
        "fixed_frame_report": result.fixed_frame_report,
    }


def pose_shape_optimization_result_to_json(result: PoseShapeOptimizationResult) -> dict[str, Any]:
    history = [dict(item) for item in result.history]
    beta_history = [dict(item) for item in result.beta_only_result.history]
    pose = np.asarray(result.pose, dtype=np.float64)
    pose_norms = np.linalg.norm(pose, axis=1)
    max_pose_idx = int(np.argmax(pose_norms))
    return {
        "side": result.side,
        "beta": [float(value) for value in result.beta],
        "pose_residual_labels": list(MANO_POSE_RESIDUAL_LABELS),
        "pose_residual": pose.tolist(),
        "pose_residual_by_joint": [
            {
                "label": MANO_POSE_RESIDUAL_LABELS[idx],
                "axis_angle": [float(value) for value in pose[idx]],
                "norm_rad": float(pose_norms[idx]),
            }
            for idx in range(len(MANO_POSE_RESIDUAL_LABELS))
        ],
        "pose_residual_max_abs_rad": float(np.max(np.abs(pose))),
        "pose_residual_max_norm": {
            "label": MANO_POSE_RESIDUAL_LABELS[max_pose_idx],
            "norm_rad": float(pose_norms[max_pose_idx]),
        },
        "robot_qpos": {name: float(value) for name, value in sorted(result.robot_qpos.items())},
        "reference_pose_fit": result.reference_pose_fit,
        "config": asdict(result.config),
        "initial_alignment_summary": result.initial_alignment_report.get("summary", {}),
        "beta_only_summary": result.beta_only_result.fixed_frame_report["summary"],
        "fixed_frame_summary": result.fixed_frame_report["summary"],
        "beta_only_final_losses": beta_history[-1] if beta_history else {},
        "final_losses": history[-1] if history else {},
        "beta_only_history": beta_history,
        "history": history,
        "fingertip_vertex_indices": dict(result.fitted_mano.fingertip_vertex_indices),
        "fixed_frame_report": result.fixed_frame_report,
    }


def write_obj(mesh: Mesh, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for vertex in np.asarray(mesh.vertices, dtype=np.float64):
            f.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in np.asarray(mesh.faces, dtype=np.int64):
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _compute_losses(
    model,
    beta,
    faces,
    face_indices,
    barycentric,
    target_surface,
    target_keypoints,
    keypoint_weights,
    *,
    step: int,
    config: ShapeOptimizationConfig,
) -> dict[str, Any]:
    vertices = model.vertices_from_beta(beta)
    mano_surface = _sample_surface_torch(vertices, faces, face_indices, barycentric)
    mano_keypoints = model.keypoints_from_vertices(vertices)

    distances = _pairwise_squared_distances(mano_surface, target_surface)
    chamfer = 0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())
    key_residual = mano_keypoints - target_keypoints
    key_loss = torch_sum(keypoint_weights * (key_residual * key_residual).sum(dim=1)) / torch_sum(keypoint_weights)
    beta_l2 = (beta * beta).mean()
    key_weight = max(0.0, 1.0 - float(step) / max(1.0, float(config.key_decay_steps)))
    total = chamfer + float(config.key_loss_weight) * key_weight * key_loss + float(config.beta_l2_weight) * beta_l2
    return {
        "total": total,
        "chamfer": chamfer,
        "key": key_loss,
        "beta_l2": beta_l2,
        "key_weight": key_weight,
    }


def _compute_pose_losses(
    model,
    beta,
    pose_raw,
    faces,
    face_indices,
    barycentric,
    target_surface,
    target_keypoints,
    keypoint_weights,
    *,
    step: int,
    config: PoseShapeOptimizationConfig,
) -> dict[str, Any]:
    pose = _bounded_pose(pose_raw, config.pose_limit_rad)
    vertices, joints = model.vertices_joints_from_beta_pose(beta, pose)
    mano_surface = _sample_surface_torch(vertices, faces, face_indices, barycentric)
    mano_keypoints = model.keypoints_from_vertices(vertices, joints)

    distances = _pairwise_squared_distances(mano_surface, target_surface)
    chamfer = 0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())
    key_residual = mano_keypoints - target_keypoints
    key_loss = torch_sum(keypoint_weights * (key_residual * key_residual).sum(dim=1)) / torch_sum(keypoint_weights)
    beta_l2 = (beta * beta).mean()
    pose_l2 = (pose * pose).mean()
    key_weight = max(0.0, 1.0 - float(step) / max(1.0, float(config.key_decay_steps)))
    total = (
        chamfer
        + float(config.key_loss_weight) * key_weight * key_loss
        + float(config.beta_l2_weight) * beta_l2
        + float(config.pose_l2_weight) * pose_l2
    )
    return {
        "total": total,
        "chamfer": chamfer,
        "key": key_loss,
        "beta_l2": beta_l2,
        "pose_l2": pose_l2,
        "pose_max_abs_rad": pose.abs().max(),
        "key_weight": key_weight,
    }


def _bounded_pose(pose_raw, pose_limit_rad: float):
    return float(pose_limit_rad) * pose_raw.tanh()


def _sample_surface_torch(vertices, faces, face_indices, barycentric):
    tri = vertices[faces[face_indices]]
    return (tri * barycentric[:, :, None]).sum(dim=1)


def _pairwise_squared_distances(a, b):
    distances = (a * a).sum(dim=1)[:, None] + (b * b).sum(dim=1)[None, :] - 2.0 * (a @ b.T)
    return distances.clamp_min(0.0)


def _losses_to_history(step: int, losses: Mapping[str, Any]) -> dict[str, float]:
    chamfer = float(losses["chamfer"].detach().cpu())
    key = float(losses["key"].detach().cpu())
    return {
        "step": float(step),
        "total": float(losses["total"].detach().cpu()),
        "chamfer": chamfer,
        "key": key,
        "beta_l2": float(losses["beta_l2"].detach().cpu()),
        "pose_l2": float(losses["pose_l2"].detach().cpu()) if "pose_l2" in losses else 0.0,
        "pose_max_abs_rad": float(losses["pose_max_abs_rad"].detach().cpu()) if "pose_max_abs_rad" in losses else 0.0,
        "key_weight": float(losses["key_weight"]),
        "chamfer_rmse_mm": float(np.sqrt(max(chamfer, 0.0)) * 1000.0),
        "key_rmse_mm": float(np.sqrt(max(key, 0.0)) * 1000.0),
    }


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


def torch_sum(value):
    return value.sum()


def _bounds(points: np.ndarray) -> dict[str, list[float]]:
    points = np.asarray(points, dtype=np.float64)
    return {
        "min": _vector(points.min(axis=0)),
        "max": _vector(points.max(axis=0)),
        "extent": _vector(points.max(axis=0) - points.min(axis=0)),
    }


def _vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


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


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when the optional dependency is unavailable.
        raise RuntimeError(
            "PyTorch is required for beta-only MANO shape optimization. "
            "Install it into the tactile_utils Conda environment, for example: "
            "conda install -n tactile_utils pytorch cpuonly -c pytorch"
        ) from exc
    return torch
