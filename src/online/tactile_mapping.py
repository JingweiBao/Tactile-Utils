from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from offline.tactile_to_mano_projection import apply_sensor_values_to_vertices, load_alignment_side_config
from sub_modules.offline_shape_alignment.alignment import apply_similarity_to_points
from sub_modules.offline_shape_alignment.mano_torch import MANOBetaModel, load_mano_beta_model
from sub_modules.offline_shape_alignment.types import KEYPOINT_LABELS, Mesh
from sub_modules.offline_shape_alignment.xhand import default_xhand_urdf_path, infer_xhand_semantic_keypoints


FINGER_KEYPOINT_LABELS: dict[str, tuple[str, str, str, str]] = {
    "thumb": ("thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip"),
    "index": ("index_mcp", "index_pip", "index_dip", "index_tip"),
    "middle": ("middle_mcp", "middle_pip", "middle_dip", "middle_tip"),
    "ring": ("ring_mcp", "ring_pip", "ring_dip", "ring_tip"),
    "pinky": ("pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"),
}
MANO_POSE_LABELS: tuple[str, ...] = (
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
MANO_FULL_POSE_LABELS: tuple[str, ...] = ("wrist",) + MANO_POSE_LABELS
MODULE_NAME = "online_tactile_mapping"


@dataclass(frozen=True)
class TactileProjectionMapping:
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    shape: tuple[int, int]
    nearest_face_indices: np.ndarray
    barycentric: np.ndarray
    valid_vertex_mask: np.ndarray
    sensor_names: tuple[str, ...]
    semantic_regions: tuple[str, ...]
    projection_mode: str
    metadata: Mapping[str, Any]

    @property
    def vertex_count(self) -> int:
        return int(self.shape[0])

    @property
    def taxel_count(self) -> int:
        return int(self.shape[1])


@dataclass(frozen=True)
class OnlineReference:
    side: str
    xhand_urdf_path: Path
    robot_to_mano: np.ndarray
    reference_qpos: dict[str, float]
    reference_keypoints_mano: dict[str, np.ndarray]
    beta: np.ndarray
    base_pose: np.ndarray
    model: MANOBetaModel
    projection: TactileProjectionMapping
    retarget_model: Any | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class OnlineTactileResult:
    side: str
    qpos: dict[str, float]
    mano_mesh: Mesh
    vertex_tactile_values: np.ndarray
    taxel_surface_points: np.ndarray
    mano_pose: np.ndarray
    mano_frame_transform: np.ndarray
    tactile_values: np.ndarray
    valid_vertex_mask: np.ndarray
    metadata: Mapping[str, Any]


def load_online_reference(
    alignment_json: str | Path,
    projection_npz: str | Path,
    side: str,
    *,
    xhand_root: str | Path = "assets/hands/xhand",
    mano_root: str | Path = "assets/hands/mano",
    xhand_urdf: str | Path | None = None,
    retarget_model_npz: str | Path | None = None,
    device: str = "cpu",
    dtype: str = "float32",
) -> OnlineReference:
    side = _validate_side(side)
    alignment_path = Path(alignment_json)
    alignment = load_alignment_side_config(alignment_path, side)
    side_payload = _alignment_side_payload(alignment_path, side)
    beta = _extract_beta(side_payload)
    base_pose = _extract_base_pose(side_payload)
    try:
        model = load_mano_beta_model(side, mano_root, num_betas=int(beta.shape[0]), device=device, dtype=dtype)
    except RuntimeError as exc:
        raise RuntimeError(
            "PyTorch is required for online MANO pose generation. "
            "Install the shape-align optional dependency or use an environment that can import torch."
        ) from exc
    projection = load_projection_mapping(projection_npz)
    retarget_model = None
    if retarget_model_npz is not None:
        from online.retarget_calibration import load_retarget_model

        retarget_model = load_retarget_model(retarget_model_npz)
        if retarget_model.side != side:
            raise ValueError(f"retarget model side {retarget_model.side!r} does not match requested side {side!r}")
    urdf_path = Path(xhand_urdf) if xhand_urdf is not None else default_xhand_urdf_path(xhand_root, side)
    reference_keypoints = infer_xhand_semantic_keypoints(urdf_path, side, qpos=alignment.robot_qpos)
    reference_keypoints_mano = _transform_keypoint_mapping(reference_keypoints.point_by_label(), alignment.robot_to_mano)

    if projection.vertex_count != int(model.v_template.shape[0]):
        raise ValueError(
            f"projection vertex count {projection.vertex_count} does not match MANO model vertex count "
            f"{int(model.v_template.shape[0])}"
        )
    if projection.nearest_face_indices.size and int(projection.nearest_face_indices.max()) >= model.faces.shape[0]:
        raise ValueError("projection nearest_face_indices reference faces outside the MANO model")

    return OnlineReference(
        side=side,
        xhand_urdf_path=urdf_path,
        robot_to_mano=alignment.robot_to_mano,
        reference_qpos=dict(alignment.robot_qpos),
        reference_keypoints_mano=reference_keypoints_mano,
        beta=beta,
        base_pose=base_pose,
        model=model,
        projection=projection,
        retarget_model=retarget_model,
        metadata={
            "alignment_json": str(alignment_path),
            "projection_npz": str(Path(projection_npz)),
            "retarget_model_npz": None if retarget_model_npz is None else str(Path(retarget_model_npz)),
            "mano_root": str(Path(mano_root)),
            "xhand_root": str(Path(xhand_root)),
            "device": device,
            "dtype": dtype,
        },
    )


def project_online_tactile(
    reference: OnlineReference,
    qpos: Mapping[str, float] | None,
    tactile_values: np.ndarray,
    *,
    mano_frame_transform: np.ndarray | None = None,
) -> OnlineTactileResult:
    merged_qpos = merge_qpos_with_reference(reference.reference_qpos, qpos)
    tactile = _coerce_tactile_values(tactile_values, reference.projection.taxel_count)
    frame_transform = _coerce_transform(mano_frame_transform)
    mano_pose = xhand_qpos_to_mano_pose(reference, merged_qpos)
    mano_instance = reference.model.instance(reference.beta, mano_pose)
    vertices = np.asarray(mano_instance.mesh.vertices, dtype=np.float64)
    faces = np.asarray(mano_instance.mesh.faces, dtype=np.int64)
    if vertices.shape[0] != reference.projection.vertex_count:
        raise ValueError(
            f"posed MANO vertex count {vertices.shape[0]} does not match projection vertex count "
            f"{reference.projection.vertex_count}"
        )
    vertices = apply_similarity_to_points(frame_transform, vertices)
    vertex_values = apply_sensor_values_to_vertices(
        reference.projection.rows,
        reference.projection.cols,
        reference.projection.vals,
        reference.projection.shape,
        tactile,
        normalize=True,
    )
    taxel_surface_points = map_taxels_to_posed_surface(
        vertices,
        faces,
        reference.projection.nearest_face_indices,
        reference.projection.barycentric,
    )
    return OnlineTactileResult(
        side=reference.side,
        qpos=merged_qpos,
        mano_mesh=Mesh(vertices=vertices, faces=faces),
        vertex_tactile_values=vertex_values,
        taxel_surface_points=taxel_surface_points,
        mano_pose=mano_pose,
        mano_frame_transform=frame_transform,
        tactile_values=tactile,
        valid_vertex_mask=reference.projection.valid_vertex_mask.copy(),
        metadata={
            "projection_mode": reference.projection.projection_mode,
            "mano_pose_labels": list(MANO_FULL_POSE_LABELS if mano_pose.shape[0] == 16 else MANO_POSE_LABELS),
            "mano_frame_transform": frame_transform.tolist(),
            "reference": dict(reference.metadata),
        },
    )


def xhand_qpos_to_mano_pose(
    reference: OnlineReference,
    qpos: Mapping[str, float] | None,
    *,
    include_root: bool = True,
) -> np.ndarray:
    merged_qpos = merge_qpos_with_reference(reference.reference_qpos, qpos)
    if reference.retarget_model is not None:
        pose = np.asarray(
            reference.retarget_model.predict(merged_qpos, reference_qpos=reference.reference_qpos),
            dtype=np.float64,
        )
        if pose.shape != (len(MANO_FULL_POSE_LABELS), 3):
            raise ValueError(f"retarget model returned pose with shape {pose.shape}")
        return pose if include_root else pose[1:]
    current_keypoints = infer_xhand_semantic_keypoints(reference.xhand_urdf_path, reference.side, qpos=merged_qpos)
    current_keypoints_mano = _transform_keypoint_mapping(current_keypoints.point_by_label(), reference.robot_to_mano)
    return keypoints_to_mano_pose(
        reference.reference_keypoints_mano,
        current_keypoints_mano,
        reference.base_pose,
        include_root=include_root,
    )


def keypoints_to_mano_pose(
    reference_keypoints_mano: Mapping[str, np.ndarray] | np.ndarray,
    current_keypoints_mano: Mapping[str, np.ndarray] | np.ndarray,
    base_pose: np.ndarray | None = None,
    *,
    include_root: bool = False,
    base_root_pose: np.ndarray | None = None,
) -> np.ndarray:
    reference = _coerce_keypoints(reference_keypoints_mano)
    current = _coerce_keypoints(current_keypoints_mano)
    base_root_vector, base_pose = _coerce_root_and_nonroot_pose(base_pose, base_root_pose)
    base_root_rotation = axis_angle_to_matrices(base_root_vector[None, :])[0]
    base_rotations = axis_angle_to_matrices(base_pose)
    pose_by_label: dict[str, np.ndarray] = {}
    reference_palm = _hand_frame(reference)
    current_palm = _hand_frame(current)
    root_rotation = (current_palm @ reference_palm.T) @ base_root_rotation

    for finger, labels in FINGER_KEYPOINT_LABELS.items():
        reference_frames = [
            _finger_segment_frame(reference[labels[idx]], reference[labels[idx + 1]], reference["wrist"])
            for idx in range(3)
        ]
        current_frames = [
            _finger_segment_frame(current[labels[idx]], current[labels[idx + 1]], current["wrist"])
            for idx in range(3)
        ]
        reference_parent = reference_palm
        current_parent = current_palm
        for segment_idx, label in enumerate(labels[:3]):
            reference_local = reference_parent.T @ reference_frames[segment_idx]
            current_local = current_parent.T @ current_frames[segment_idx]
            pose_idx = MANO_POSE_LABELS.index(label)
            dynamic_delta = current_local @ reference_local.T
            pose_by_label[label] = matrix_to_axis_angle(dynamic_delta @ base_rotations[pose_idx])
            reference_parent = reference_frames[segment_idx]
            current_parent = current_frames[segment_idx]

    nonroot_pose = np.asarray([pose_by_label[label] for label in MANO_POSE_LABELS], dtype=np.float64)
    if include_root:
        root_pose = matrix_to_axis_angle(root_rotation)[None, :]
        return np.concatenate([root_pose, nonroot_pose], axis=0)
    return nonroot_pose


def merge_qpos_with_reference(
    reference_qpos: Mapping[str, float],
    qpos: Mapping[str, float] | None,
) -> dict[str, float]:
    merged = {str(name): float(value) for name, value in reference_qpos.items()}
    for name, value in dict(qpos or {}).items():
        merged[str(name)] = float(value)
    return merged


def load_projection_mapping(path: str | Path) -> TactileProjectionMapping:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        rows = np.asarray(payload["rows"], dtype=np.int64)
        cols = np.asarray(payload["cols"], dtype=np.int64)
        vals = np.asarray(payload["vals"], dtype=np.float64)
        shape_array = np.asarray(payload["shape"], dtype=np.int64)
        if shape_array.shape != (2,):
            raise ValueError(f"projection shape must have shape (2,), got {shape_array.shape}")
        shape = (int(shape_array[0]), int(shape_array[1]))
        nearest_face_indices = np.asarray(payload["nearest_face_indices"], dtype=np.int64)
        barycentric = np.asarray(payload["barycentric"], dtype=np.float64)
        valid_vertex_mask = (
            np.asarray(payload["valid_vertex_mask"], dtype=np.bool_)
            if "valid_vertex_mask" in payload.files
            else _valid_mask_from_sparse(rows, vals, shape[0])
        )
        sensor_names = tuple(str(item) for item in np.asarray(payload["sensor_names"]).tolist()) if "sensor_names" in payload.files else ()
        semantic_regions = (
            tuple(str(item) for item in np.asarray(payload["semantic_regions"]).tolist())
            if "semantic_regions" in payload.files
            else ()
        )
        projection_mode = str(np.asarray(payload["projection_mode"]).item()) if "projection_mode" in payload.files else "unknown"

    _validate_projection_arrays(rows, cols, vals, shape, nearest_face_indices, barycentric, valid_vertex_mask)
    return TactileProjectionMapping(
        rows=rows,
        cols=cols,
        vals=vals,
        shape=shape,
        nearest_face_indices=nearest_face_indices,
        barycentric=barycentric,
        valid_vertex_mask=valid_vertex_mask,
        sensor_names=sensor_names,
        semantic_regions=semantic_regions,
        projection_mode=projection_mode,
        metadata={"projection_npz": str(path)},
    )


def map_taxels_to_posed_surface(
    mano_vertices: np.ndarray,
    mano_faces: np.ndarray,
    nearest_face_indices: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(mano_vertices, dtype=np.float64)
    faces = np.asarray(mano_faces, dtype=np.int64)
    face_indices = np.asarray(nearest_face_indices, dtype=np.int64)
    bary = np.asarray(barycentric, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"mano_vertices must have shape (V, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"mano_faces must have shape (F, 3), got {faces.shape}")
    if face_indices.ndim != 1:
        raise ValueError(f"nearest_face_indices must have shape (T,), got {face_indices.shape}")
    if bary.shape != (face_indices.shape[0], 3):
        raise ValueError(f"barycentric must have shape ({face_indices.shape[0]}, 3), got {bary.shape}")
    if face_indices.size and (int(face_indices.min()) < 0 or int(face_indices.max()) >= faces.shape[0]):
        raise ValueError("nearest_face_indices reference faces outside mano_faces")
    triangles = vertices[faces[face_indices]]
    return np.einsum("tvc,tv->tc", triangles, bary)


def save_online_result_npz(result: OnlineTactileResult, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qpos_names = np.asarray(sorted(result.qpos))
    qpos_values = np.asarray([result.qpos[name] for name in qpos_names], dtype=np.float64)
    np.savez_compressed(
        out_path,
        mano_vertices=result.mano_mesh.vertices.astype(np.float64),
        mano_faces=result.mano_mesh.faces.astype(np.int64),
        vertex_tactile_values=result.vertex_tactile_values.astype(np.float64),
        taxel_surface_points=result.taxel_surface_points.astype(np.float64),
        mano_pose=result.mano_pose.astype(np.float64),
        mano_frame_transform=result.mano_frame_transform.astype(np.float64),
        tactile_values=result.tactile_values.astype(np.float64),
        valid_vertex_mask=result.valid_vertex_mask.astype(np.bool_),
        qpos_names=qpos_names,
        qpos_values=qpos_values,
    )


def render_online_tactile_map(
    result: OnlineTactileResult,
    out_path: str | Path,
    *,
    width: int = 900,
    height: int = 700,
    title: str | None = None,
) -> None:
    from PIL import Image, ImageDraw

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    margin = 44
    bounds_points = np.concatenate([result.mano_mesh.vertices, result.taxel_surface_points], axis=0)
    projected_bounds, _ = _project_view(bounds_points, result.side)
    xy_min = projected_bounds.min(axis=0)
    xy_max = projected_bounds.max(axis=0)
    extent = np.maximum(xy_max - xy_min, 1e-9)
    scale = min((width - 2 * margin) / extent[0], (height - 2 * margin) / extent[1])
    center = (xy_min + xy_max) / 2.0

    def to_screen(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy, z = _project_view(points, result.side)
        screen = np.empty_like(xy)
        screen[:, 0] = (xy[:, 0] - center[0]) * scale + width / 2.0
        screen[:, 1] = height / 2.0 - (xy[:, 1] - center[1]) * scale
        return screen, z

    vertices_xy, vertex_depth = to_screen(result.mano_mesh.vertices)
    face_depth = vertex_depth[result.mano_mesh.faces].mean(axis=1)
    for face_idx in np.argsort(face_depth):
        face = result.mano_mesh.faces[int(face_idx)]
        polygon = [(float(x), float(y)) for x, y in vertices_xy[face]]
        shade = _face_shade(result.mano_mesh.vertices[face], result.side)
        draw.polygon(polygon, fill=(shade, shade, shade, 62), outline=(92, 92, 92, 30))

    tactile_xy, _ = to_screen(result.taxel_surface_points)
    colors = _value_colors(result.tactile_values)
    for xy, color in zip(tactile_xy, colors):
        radius = 2
        draw.ellipse(
            (float(xy[0] - radius), float(xy[1] - radius), float(xy[0] + radius), float(xy[1] + radius)),
            fill=tuple(color) + (235,),
            outline=(255, 255, 255, 185),
        )

    tactile = result.tactile_values
    draw.text(
        (14, 12),
        title or f"{result.side} online tactile map on posed MANO",
        fill=(20, 20, 20, 255),
    )
    draw.text(
        (14, height - 30),
        f"taxels={tactile.shape[0]} vertices={result.mano_mesh.vertices.shape[0]} "
        f"valid_vertices={int(result.valid_vertex_mask.sum())} "
        f"tactile min/mean/max={float(tactile.min()):.3g}/{float(tactile.mean()):.3g}/{float(tactile.max()):.3g}",
        fill=(75, 75, 75, 255),
    )
    image.convert("RGB").save(out_path)


def online_result_to_json(result: OnlineTactileResult, outputs: Mapping[str, str] | None = None) -> dict[str, Any]:
    pose_norms = np.linalg.norm(result.mano_pose, axis=1)
    tactile = result.tactile_values
    return {
        "module": MODULE_NAME,
        "side": result.side,
        "taxel_count": int(tactile.shape[0]),
        "mano_vertex_count": int(result.mano_mesh.vertices.shape[0]),
        "mano_face_count": int(result.mano_mesh.faces.shape[0]),
        "valid_vertex_count": int(result.valid_vertex_mask.sum()),
        "mano_pose_norm_rad": {
            "mean": float(pose_norms.mean()) if pose_norms.size else 0.0,
            "max": float(pose_norms.max()) if pose_norms.size else 0.0,
        },
        "mano_pose_shape": list(result.mano_pose.shape),
        "mano_pose_labels": list(MANO_FULL_POSE_LABELS if result.mano_pose.shape[0] == 16 else MANO_POSE_LABELS),
        "mano_frame_transform": result.mano_frame_transform.tolist(),
        "tactile_values": {
            "min": float(tactile.min()) if tactile.size else 0.0,
            "mean": float(tactile.mean()) if tactile.size else 0.0,
            "max": float(tactile.max()) if tactile.size else 0.0,
            "nonzero_count": int(np.count_nonzero(tactile)),
        },
        "qpos": {name: float(value) for name, value in sorted(result.qpos.items())},
        "outputs": dict(outputs or {}),
        "metadata": _jsonable(result.metadata),
    }


def axis_angle_to_matrices(axis_angle: np.ndarray) -> np.ndarray:
    values = np.asarray(axis_angle, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"axis_angle must have shape (N, 3), got {values.shape}")
    matrices = np.empty((values.shape[0], 3, 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for idx, vector in enumerate(values):
        theta = float(np.linalg.norm(vector))
        if theta <= 1e-12:
            skew = _skew(vector)
            matrices[idx] = eye + skew
            continue
        axis = vector / theta
        skew = _skew(axis)
        matrices[idx] = eye + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    return matrices


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3), got {rotation.shape}")
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta <= 1e-8:
        return 0.5 * np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
    sin_theta = float(np.sin(theta))
    if abs(sin_theta) > 1e-8:
        axis = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * sin_theta)
        return axis * theta
    eigenvalues, eigenvectors = np.linalg.eig(rotation)
    axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
    axis = _normalize_vector(axis)
    return axis * theta


def load_json_qpos(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, Mapping) and isinstance(payload.get("qpos"), Mapping):
        payload = payload["qpos"]
    if not isinstance(payload, Mapping):
        raise ValueError("qpos JSON must be a mapping or contain a mapping under key 'qpos'")
    return {str(name): float(value) for name, value in payload.items()}


def load_tactile_values(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, Mapping):
        payload = payload.get("tactile_values")
    return np.asarray(payload, dtype=np.float64)


def _alignment_side_payload(path: Path, side: str) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    side_payload = payload.get("sides", {}).get(side)
    if not isinstance(side_payload, Mapping):
        raise ValueError(f"{path} does not contain side {side!r}")
    return side_payload


def _extract_beta(side_payload: Mapping[str, Any]) -> np.ndarray:
    if "beta" not in side_payload:
        raise ValueError("alignment JSON side payload must contain fitted MANO beta")
    beta = np.asarray(side_payload["beta"], dtype=np.float64)
    if beta.ndim != 1 or beta.size == 0:
        raise ValueError(f"beta must be a non-empty 1D array, got {beta.shape}")
    return beta


def _extract_base_pose(side_payload: Mapping[str, Any]) -> np.ndarray:
    pose = np.asarray(side_payload.get("pose_residual", np.zeros((len(MANO_POSE_LABELS), 3))), dtype=np.float64)
    if pose.shape != (len(MANO_POSE_LABELS), 3):
        raise ValueError(f"pose_residual must have shape ({len(MANO_POSE_LABELS)}, 3), got {pose.shape}")
    labels = tuple(side_payload.get("pose_residual_labels", MANO_POSE_LABELS))
    if labels != MANO_POSE_LABELS:
        raise ValueError(f"unsupported pose_residual_labels order: {labels!r}")
    return pose


def _transform_keypoint_mapping(keypoints: Mapping[str, np.ndarray], transform: np.ndarray) -> dict[str, np.ndarray]:
    labels = list(keypoints)
    points = np.asarray([keypoints[label] for label in labels], dtype=np.float64)
    transformed = apply_similarity_to_points(np.asarray(transform, dtype=np.float64), points)
    return {label: transformed[idx] for idx, label in enumerate(labels)}


def _coerce_keypoints(value: Mapping[str, np.ndarray] | np.ndarray) -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        out = {str(label): np.asarray(point, dtype=np.float64) for label, point in value.items()}
    else:
        points = np.asarray(value, dtype=np.float64)
        if points.shape != (len(KEYPOINT_LABELS), 3):
            raise ValueError(f"keypoint array must have shape ({len(KEYPOINT_LABELS)}, 3), got {points.shape}")
        out = {label: points[idx] for idx, label in enumerate(KEYPOINT_LABELS)}
    missing = sorted(set(KEYPOINT_LABELS) - set(out))
    if missing:
        raise ValueError(f"keypoints missing labels: {missing}")
    for label, point in out.items():
        if point.shape != (3,):
            raise ValueError(f"keypoint {label!r} must have shape (3,), got {point.shape}")
    return out


def _coerce_root_and_nonroot_pose(
    pose: np.ndarray | None,
    root_pose: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if pose is None:
        nonroot = np.zeros((len(MANO_POSE_LABELS), 3), dtype=np.float64)
        root = np.zeros(3, dtype=np.float64)
    else:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape == (len(MANO_FULL_POSE_LABELS), 3):
            root = pose[0].copy()
            nonroot = pose[1:].copy()
        elif pose.shape == (len(MANO_POSE_LABELS), 3):
            root = np.zeros(3, dtype=np.float64)
            nonroot = pose.copy()
        else:
            raise ValueError(
                f"pose must have shape ({len(MANO_POSE_LABELS)}, 3) or ({len(MANO_FULL_POSE_LABELS)}, 3), "
                f"got {pose.shape}"
            )
    if root_pose is not None:
        root = np.asarray(root_pose, dtype=np.float64)
        if root.shape != (3,):
            raise ValueError(f"root_pose must have shape (3,), got {root.shape}")
    return root, nonroot


def _coerce_pose(pose: np.ndarray | None) -> np.ndarray:
    _root, nonroot = _coerce_root_and_nonroot_pose(pose, None)
    return nonroot


def _coerce_transform(transform: np.ndarray | None) -> np.ndarray:
    if transform is None:
        return np.eye(4, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"mano_frame_transform must have shape (4, 4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError("mano_frame_transform must contain finite values")
    return transform


def _coerce_tactile_values(values: np.ndarray, expected_count: int) -> np.ndarray:
    tactile = np.asarray(values, dtype=np.float64)
    if tactile.shape != (expected_count,):
        raise ValueError(f"tactile_values must have shape ({expected_count},), got {tactile.shape}")
    return tactile


def _validate_projection_arrays(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    shape: tuple[int, int],
    nearest_face_indices: np.ndarray,
    barycentric: np.ndarray,
    valid_vertex_mask: np.ndarray,
) -> None:
    if rows.shape != cols.shape or rows.shape != vals.shape:
        raise ValueError("rows, cols, and vals must have matching shapes")
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"projection shape must be positive, got {shape}")
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= shape[0]):
        raise ValueError("projection rows are outside shape")
    if cols.size and (int(cols.min()) < 0 or int(cols.max()) >= shape[1]):
        raise ValueError("projection cols are outside shape")
    if nearest_face_indices.shape != (shape[1],):
        raise ValueError(f"nearest_face_indices must have shape ({shape[1]},), got {nearest_face_indices.shape}")
    if barycentric.shape != (shape[1], 3):
        raise ValueError(f"barycentric must have shape ({shape[1]}, 3), got {barycentric.shape}")
    if valid_vertex_mask.shape != (shape[0],):
        raise ValueError(f"valid_vertex_mask must have shape ({shape[0]},), got {valid_vertex_mask.shape}")


def _valid_mask_from_sparse(rows: np.ndarray, vals: np.ndarray, vertex_count: int) -> np.ndarray:
    weights = np.zeros(vertex_count, dtype=np.float64)
    np.add.at(weights, rows, vals)
    return weights > 0.0


def _hand_frame(keypoints: Mapping[str, np.ndarray]) -> np.ndarray:
    wrist = keypoints["wrist"]
    distal = _normalize_vector(keypoints["middle_mcp"] - wrist)
    lateral = _normalize_vector(keypoints["index_mcp"] - keypoints["pinky_mcp"])
    normal = np.cross(lateral, distal)
    if float(np.linalg.norm(normal)) <= 1e-9:
        normal = _fallback_perpendicular(distal)
    else:
        normal = _normalize_vector(normal)
    lateral = _normalize_vector(np.cross(distal, normal))
    return np.stack([lateral, distal, normal], axis=1)


def _finger_segment_frame(start: np.ndarray, end: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    tangent = _normalize_vector(np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64))
    reference = ((np.asarray(start, dtype=np.float64) + np.asarray(end, dtype=np.float64)) * 0.5) - np.asarray(
        wrist,
        dtype=np.float64,
    )
    side_axis = reference - tangent * float(reference @ tangent)
    if float(np.linalg.norm(side_axis)) <= 1e-9:
        side_axis = _fallback_perpendicular(tangent)
    else:
        side_axis = _normalize_vector(side_axis)
    normal_axis = _normalize_vector(np.cross(tangent, side_axis))
    side_axis = _normalize_vector(np.cross(normal_axis, tangent))
    return np.stack([tangent, side_axis, normal_axis], axis=1)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-18:
        raise ValueError("cannot normalize a degenerate vector")
    return vector / norm


def _fallback_perpendicular(tangent: np.ndarray) -> np.ndarray:
    axes = np.eye(3, dtype=np.float64)
    axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
    return _normalize_vector(axis - tangent * float(axis @ tangent))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def _project_view(points: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    view_sign = -1.0 if side == "left" else 1.0
    x = points[:, 0] + 0.85 * view_sign * points[:, 1]
    y = points[:, 2] - 0.12 * view_sign * points[:, 1]
    depth = view_sign * points[:, 1]
    return np.stack([x, y], axis=1), depth


def _face_shade(triangle: np.ndarray, side: str) -> int:
    view = np.asarray([0.0, -1.0 if side == "left" else 1.0, 0.25], dtype=np.float64)
    view /= np.linalg.norm(view)
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-18:
        return 190
    normal /= norm
    light = max(0.0, float(np.dot(normal, view)))
    return int(176 + 46 * light)


def _value_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        scaled = np.zeros_like(values)
    else:
        low = float(finite.min())
        high = float(finite.max())
        if abs(high - low) <= 1e-18:
            scaled = np.full_like(values, 0.5, dtype=np.float64)
        else:
            scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    blue = np.asarray([39, 118, 214], dtype=np.float64)
    yellow = np.asarray([244, 196, 48], dtype=np.float64)
    red = np.asarray([225, 78, 54], dtype=np.float64)
    colors = np.empty((values.shape[0], 3), dtype=np.float64)
    lower = scaled <= 0.5
    colors[lower] = blue + (yellow - blue) * (scaled[lower, None] * 2.0)
    colors[~lower] = yellow + (red - yellow) * ((scaled[~lower, None] - 0.5) * 2.0)
    return np.clip(colors, 0, 255).astype(np.uint8)


def _validate_side(side: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    return side


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
