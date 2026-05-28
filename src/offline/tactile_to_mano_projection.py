from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sub_modules.offline_shape_alignment.alignment import apply_similarity_to_points
from sub_modules.offline_shape_alignment.types import Mesh
from sub_modules.tactile_layout_3d_projection.geometry import transform_points
from sub_modules.tactile_layout_3d_projection.inspire import infer_urdf_root_link
from sub_modules.tactile_layout_3d_projection.urdf import load_urdf
from sub_modules.tactile_layout_3d_projection.xhand import (
    FINGER_LINKS,
    _load_tactile_xml_points,
    _mujoco_tactile_points_to_urdf_local,
)


PROJECTION_MODES = (
    "global-nearest",
    "semantic-distal",
    "semantic-normalized",
    "block-preserving",
    "graph-preserving",
    "graph-preserving-no-block",
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_ALIASES = {"mid": "middle"}
FINGER_KEYPOINT_LABELS = {
    "thumb": ("thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip"),
    "index": ("index_mcp", "index_pip", "index_dip", "index_tip"),
    "middle": ("middle_mcp", "middle_pip", "middle_dip", "middle_tip"),
    "ring": ("ring_mcp", "ring_pip", "ring_dip", "ring_tip"),
    "pinky": ("pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"),
}
SEMANTIC_DISTANCE_WARN_DELTA_M = 0.015
SEMANTIC_DISTANCE_WARN_ABS_M = 0.020
BLOCK_PRESERVING_SCALE_RANGE = (0.5, 2.0)
BLOCK_PRESERVING_PCA_MIN_SINGULAR = 1e-9
GRAPH_PRESERVING_K = 4
GRAPH_PRESERVING_ITERATIONS = 8
GRAPH_PRESERVING_STEP_SIZE = 0.18
GRAPH_PRESERVING_DATA_WEIGHT = 0.35
GRAPH_PRESERVING_EDGE_WEIGHT = 0.55
GRAPH_PRESERVING_LAPLACIAN_WEIGHT = 0.10
COLLAPSE_DISTANCE_THRESHOLD_M = 0.001


@dataclass(frozen=True)
class TactileLayout:
    side: str
    points_robot: np.ndarray
    sensor_names: tuple[str, ...]
    semantic_regions: tuple[str, ...]
    taxel_indices: np.ndarray
    sensor_slices: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class AlignmentSideConfig:
    side: str
    robot_to_mano: np.ndarray
    robot_qpos: dict[str, float]
    mano_keypoints: dict[str, np.ndarray]
    fitted_mano_obj: Path | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ManoSemanticRegions:
    finger_faces: dict[str, np.ndarray]
    distal_faces: dict[str, np.ndarray]
    vertex_finger: np.ndarray
    vertex_path_fraction: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ProjectionResult:
    side: str
    projection_mode: str
    layout: TactileLayout
    mano_mesh: Mesh
    robot_to_mano: np.ndarray
    tactile_points_mano: np.ndarray
    projected_points: np.ndarray
    nearest_face_indices: np.ndarray
    barycentric: np.ndarray
    distances_m: np.ndarray
    global_projected_points: np.ndarray
    global_nearest_face_indices: np.ndarray
    global_barycentric: np.ndarray
    global_distances_m: np.ndarray
    target_fingers: tuple[str | None, ...]
    candidate_face_counts: np.ndarray
    fallback_reasons: tuple[str, ...]
    semantic_warning_mask: np.ndarray
    semantic_target_points: np.ndarray
    normalized_coordinates: np.ndarray
    block_ids: tuple[str, ...]
    block_local_uv: np.ndarray
    block_target_points: np.ndarray
    block_similarity_scale: np.ndarray
    block_fallback_reasons: tuple[str, ...]
    block_layout_preservation: np.ndarray
    graph_knn_indices: np.ndarray
    graph_edge_length_error: np.ndarray
    graph_neighbor_mismatch: np.ndarray
    graph_angle_error_rad: np.ndarray
    graph_laplacian_error: np.ndarray
    graph_fallback_reasons: tuple[str, ...]
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    shape: tuple[int, int]
    vertex_weight_sum: np.ndarray
    valid_vertex_mask: np.ndarray
    metadata: Mapping[str, Any]


def build_xhand_tactile_layout(
    urdf_path: str | Path,
    tactile_dir: str | Path,
    side: str,
    *,
    qpos: Mapping[str, float] | None = None,
) -> TactileLayout:
    side = _validate_side(side)
    urdf_path = Path(urdf_path)
    tactile_dir = Path(tactile_dir)
    model = load_urdf(urdf_path)
    root_link = infer_urdf_root_link(model.links, model.joints)
    link_transforms = model.link_transforms(dict(qpos or {}), root_link=root_link)

    all_points: list[np.ndarray] = []
    sensor_names: list[str] = []
    semantic_regions: list[str] = []
    taxel_indices: list[np.ndarray] = []
    sensor_slices: list[dict[str, Any]] = []
    offset = 0

    for finger, link_template in FINGER_LINKS.items():
        link_name = link_template.format(side=side)
        if link_name not in link_transforms:
            raise ValueError(f"{side} XHand URDF missing tactile parent link {link_name!r}")
        local_mujoco = _load_tactile_xml_points(tactile_dir / f"{finger}_collisions.xml")
        local_urdf = _mujoco_tactile_points_to_urdf_local(local_mujoco, finger, side)
        points_robot = transform_points(link_transforms[link_name], local_urdf)
        count = int(points_robot.shape[0])
        name = f"{side}_{finger}_tactile_sensor"
        semantic_region = f"{side}_{finger}"
        indices = np.stack(
            [np.arange(count, dtype=np.int64), np.zeros(count, dtype=np.int64)],
            axis=1,
        )

        all_points.append(points_robot)
        sensor_names.extend([name] * count)
        semantic_regions.extend([semantic_region] * count)
        taxel_indices.append(indices)
        sensor_slices.append(
            {
                "name": name,
                "semantic_region": semantic_region,
                "link_name": link_name,
                "start": offset,
                "count": count,
            }
        )
        offset += count

    points = np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3), dtype=np.float64)
    indices = np.concatenate(taxel_indices, axis=0) if taxel_indices else np.zeros((0, 2), dtype=np.int64)
    return TactileLayout(
        side=side,
        points_robot=points,
        sensor_names=tuple(sensor_names),
        semantic_regions=tuple(semantic_regions),
        taxel_indices=indices,
        sensor_slices=tuple(sensor_slices),
        metadata={
            "hand": "xhand",
            "side": side,
            "urdf_path": str(urdf_path),
            "tactile_dir": str(tactile_dir),
            "root_link": root_link,
            "qpos": {name: float(value) for name, value in sorted(dict(qpos or {}).items())},
        },
    )


def project_tactile_layout_to_mano(
    layout: TactileLayout,
    mano_mesh: Mesh,
    robot_to_mano: np.ndarray,
    *,
    projection_mode: str = "global-nearest",
    mano_keypoints: Mapping[str, Any] | np.ndarray | None = None,
    robot_keypoints: Mapping[str, Any] | np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProjectionResult:
    projection_mode = _validate_projection_mode(projection_mode)
    robot_to_mano = np.asarray(robot_to_mano, dtype=np.float64)
    if robot_to_mano.shape != (4, 4):
        raise ValueError(f"robot_to_mano must have shape (4, 4), got {robot_to_mano.shape}")
    vertices = np.asarray(mano_mesh.vertices, dtype=np.float64)
    faces = np.asarray(mano_mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"MANO vertices must have shape (V, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"MANO faces must have shape (F, 3), got {faces.shape}")

    tactile_points_mano = apply_similarity_to_points(robot_to_mano, layout.points_robot)
    global_faces, global_barycentric, global_projected, global_distances = project_points_to_mesh(
        tactile_points_mano,
        mano_mesh,
    )
    taxel_count = tactile_points_mano.shape[0]
    block_ids = tuple(layout.sensor_names)
    block_local_uv = _empty_block_local_uv(taxel_count)
    block_target_points = np.full_like(tactile_points_mano, np.nan)
    block_similarity_scale = np.full(taxel_count, np.nan, dtype=np.float64)
    block_fallback_reasons = tuple("not_applicable" for _ in range(taxel_count))
    block_layout_preservation = np.full(taxel_count, np.nan, dtype=np.float64)
    graph_knn_indices = _empty_graph_knn_indices(taxel_count)
    graph_edge_length_error = np.full(taxel_count, np.nan, dtype=np.float64)
    graph_neighbor_mismatch = np.full(taxel_count, np.nan, dtype=np.float64)
    graph_angle_error_rad = np.full(taxel_count, np.nan, dtype=np.float64)
    graph_laplacian_error = np.full(taxel_count, np.nan, dtype=np.float64)
    graph_fallback_reasons = tuple("not_applicable" for _ in range(taxel_count))

    if projection_mode == "global-nearest":
        nearest_faces = global_faces
        barycentric = global_barycentric
        projected = global_projected
        distances = global_distances
        target_fingers = tuple(
            infer_finger_from_name(region, fallback=name)
            for region, name in zip(layout.semantic_regions, layout.sensor_names)
        )
        candidate_face_counts = np.full(taxel_count, faces.shape[0], dtype=np.int64)
        fallback_reasons = tuple("not_applicable" for _ in range(taxel_count))
        semantic_warning_mask = np.zeros(taxel_count, dtype=np.bool_)
        semantic_target_points = tactile_points_mano.copy()
        normalized_coordinates = _empty_normalized_coordinates(taxel_count)
        semantic_metadata: dict[str, Any] = {}
    else:
        if mano_keypoints is None:
            raise ValueError(f"{projection_mode} projection requires MANO 21 keypoints")
        mano_keypoints_dict = _coerce_mano_keypoints(mano_keypoints)
        regions = build_mano_semantic_regions(mano_mesh, mano_keypoints_dict)
        distal_result = _project_points_semantic_distal(
            tactile_points_mano,
            layout,
            mano_mesh,
            regions,
            global_faces,
            global_barycentric,
            global_projected,
            global_distances,
        )
        if projection_mode == "semantic-distal":
            (
                nearest_faces,
                barycentric,
                projected,
                distances,
                target_fingers,
                candidate_face_counts,
                fallback_reasons,
                semantic_warning_mask,
                semantic_target_points,
                normalized_coordinates,
            ) = distal_result
            fallback_chain = ["semantic-distal", "global-nearest"]
        else:
            if robot_keypoints is None:
                raise ValueError(f"{projection_mode} projection requires XHand/robot 21 keypoints")
            robot_keypoints_dict = _coerce_mano_keypoints(robot_keypoints)
            normalized_result = _project_points_semantic_normalized(
                layout.points_robot,
                tactile_points_mano,
                layout,
                mano_mesh,
                regions,
                robot_keypoints_dict,
                mano_keypoints_dict,
                distal_result,
                global_distances,
            )
            if projection_mode == "semantic-normalized":
                (
                    nearest_faces,
                    barycentric,
                    projected,
                    distances,
                    target_fingers,
                    candidate_face_counts,
                    fallback_reasons,
                    semantic_warning_mask,
                    semantic_target_points,
                    normalized_coordinates,
                ) = normalized_result
                fallback_chain = ["semantic-normalized", "semantic-distal", "global-nearest"]
            elif projection_mode == "graph-preserving-no-block":
                (
                    nearest_faces,
                    barycentric,
                    projected,
                    distances,
                    target_fingers,
                    candidate_face_counts,
                    fallback_reasons,
                    semantic_warning_mask,
                    semantic_target_points,
                    normalized_coordinates,
                    block_ids,
                    block_local_uv,
                    block_target_points,
                    block_similarity_scale,
                    block_fallback_reasons,
                    block_layout_preservation,
                    graph_knn_indices,
                    graph_edge_length_error,
                    graph_neighbor_mismatch,
                    graph_angle_error_rad,
                    graph_laplacian_error,
                    graph_fallback_reasons,
                ) = _project_points_graph_preserving_no_block(
                    tactile_points_mano,
                    layout,
                    mano_mesh,
                    regions,
                    robot_to_mano,
                    robot_keypoints_dict,
                    normalized_result,
                    global_distances,
                )
                fallback_chain = [
                    "graph-preserving-no-block",
                    "semantic-normalized",
                    "semantic-distal",
                    "global-nearest",
                ]
            else:
                block_result = _project_points_block_preserving(
                    layout.points_robot,
                    tactile_points_mano,
                    layout,
                    mano_mesh,
                    regions,
                    robot_to_mano,
                    robot_keypoints_dict,
                    mano_keypoints_dict,
                    normalized_result,
                    global_distances,
                )
                if projection_mode == "block-preserving":
                    (
                        nearest_faces,
                        barycentric,
                        projected,
                        distances,
                        target_fingers,
                        candidate_face_counts,
                        fallback_reasons,
                        semantic_warning_mask,
                        semantic_target_points,
                        normalized_coordinates,
                        block_ids,
                        block_local_uv,
                        block_target_points,
                        block_similarity_scale,
                        block_fallback_reasons,
                        block_layout_preservation,
                    ) = block_result
                    fallback_chain = [
                        "block-preserving",
                        "semantic-normalized",
                        "semantic-distal",
                        "global-nearest",
                    ]
                else:
                    (
                        nearest_faces,
                        barycentric,
                        projected,
                        distances,
                        target_fingers,
                        candidate_face_counts,
                        fallback_reasons,
                        semantic_warning_mask,
                        semantic_target_points,
                        normalized_coordinates,
                        block_ids,
                        block_local_uv,
                        block_target_points,
                        block_similarity_scale,
                        block_fallback_reasons,
                        block_layout_preservation,
                        graph_knn_indices,
                        graph_edge_length_error,
                        graph_neighbor_mismatch,
                        graph_angle_error_rad,
                        graph_laplacian_error,
                        graph_fallback_reasons,
                    ) = _project_points_graph_preserving(
                        layout,
                        tactile_points_mano,
                        mano_mesh,
                        regions,
                        block_result,
                        global_distances,
                    )
                    fallback_chain = [
                        "graph-preserving",
                        "block-preserving",
                        "semantic-normalized",
                        "semantic-distal",
                        "global-nearest",
                    ]
        semantic_metadata = {
            "mano_semantic_regions": _jsonable(regions.metadata),
            "projection_fallback_chain": fallback_chain,
        }
    if projection_mode not in {"block-preserving", "graph-preserving", "graph-preserving-no-block"}:
        block_target_points = semantic_target_points.copy()
    face_vertices = faces[nearest_faces]
    rows = face_vertices.reshape(-1).astype(np.int64)
    cols = np.repeat(np.arange(tactile_points_mano.shape[0], dtype=np.int64), 3)
    vals = barycentric.reshape(-1).astype(np.float64)
    shape = (int(vertices.shape[0]), int(tactile_points_mano.shape[0]))
    vertex_weight_sum = np.zeros(shape[0], dtype=np.float64)
    np.add.at(vertex_weight_sum, rows, vals)
    valid_vertex_mask = vertex_weight_sum > 0.0

    return ProjectionResult(
        side=layout.side,
        projection_mode=projection_mode,
        layout=layout,
        mano_mesh=Mesh(vertices=vertices, faces=faces),
        robot_to_mano=robot_to_mano,
        tactile_points_mano=tactile_points_mano,
        projected_points=projected,
        nearest_face_indices=nearest_faces,
        barycentric=barycentric,
        distances_m=distances,
        global_projected_points=global_projected,
        global_nearest_face_indices=global_faces,
        global_barycentric=global_barycentric,
        global_distances_m=global_distances,
        target_fingers=target_fingers,
        candidate_face_counts=candidate_face_counts,
        fallback_reasons=fallback_reasons,
        semantic_warning_mask=semantic_warning_mask,
        semantic_target_points=semantic_target_points,
        normalized_coordinates=normalized_coordinates,
        block_ids=block_ids,
        block_local_uv=block_local_uv,
        block_target_points=block_target_points,
        block_similarity_scale=block_similarity_scale,
        block_fallback_reasons=block_fallback_reasons,
        block_layout_preservation=block_layout_preservation,
        graph_knn_indices=graph_knn_indices,
        graph_edge_length_error=graph_edge_length_error,
        graph_neighbor_mismatch=graph_neighbor_mismatch,
        graph_angle_error_rad=graph_angle_error_rad,
        graph_laplacian_error=graph_laplacian_error,
        graph_fallback_reasons=graph_fallback_reasons,
        rows=rows,
        cols=cols,
        vals=vals,
        shape=shape,
        vertex_weight_sum=vertex_weight_sum,
        valid_vertex_mask=valid_vertex_mask,
        metadata={**dict(metadata or {}), **semantic_metadata},
    )


def project_points_to_mesh(
    points: np.ndarray,
    mesh: Mesh,
    candidate_face_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if faces.size == 0:
        raise ValueError("mesh has no faces")
    if candidate_face_indices is None:
        candidate_face_indices = np.arange(faces.shape[0], dtype=np.int64)
    else:
        candidate_face_indices = np.asarray(candidate_face_indices, dtype=np.int64)
    if candidate_face_indices.size == 0:
        raise ValueError("candidate_face_indices must not be empty")
    triangles = vertices[faces[candidate_face_indices]]

    nearest_faces = np.empty(points.shape[0], dtype=np.int64)
    barycentric = np.empty((points.shape[0], 3), dtype=np.float64)
    projected = np.empty_like(points)
    distances = np.empty(points.shape[0], dtype=np.float64)
    for idx, point in enumerate(points):
        closest, bary = closest_points_on_triangles(point, triangles)
        diff = closest - point[None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        face_idx = int(np.argmin(dist2))
        nearest_faces[idx] = int(candidate_face_indices[face_idx])
        barycentric[idx] = _normalize_barycentric(bary[face_idx])
        projected[idx] = closest[face_idx]
        distances[idx] = float(np.sqrt(dist2[face_idx]))
    return nearest_faces, barycentric, projected, distances


def build_mano_semantic_regions(
    mesh: Mesh,
    mano_keypoints: Mapping[str, Any] | np.ndarray,
) -> ManoSemanticRegions:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    keypoints = _coerce_mano_keypoints(mano_keypoints)
    best_distances = np.full(vertices.shape[0], np.inf, dtype=np.float64)
    vertex_finger = np.full(vertices.shape[0], "", dtype=object)
    vertex_path_fraction = np.zeros(vertices.shape[0], dtype=np.float64)
    distal_start_fraction: dict[str, float] = {}

    for finger in FINGER_NAMES:
        joints = np.asarray([keypoints[label] for label in FINGER_KEYPOINT_LABELS[finger]], dtype=np.float64)
        distances, fractions = _point_polyline_distances(vertices, joints)
        update = distances < best_distances
        best_distances[update] = distances[update]
        vertex_finger[update] = finger
        vertex_path_fraction[update] = fractions[update]
        distal_start_fraction[finger] = _distal_start_fraction(joints)

    finger_faces: dict[str, np.ndarray] = {}
    distal_faces: dict[str, np.ndarray] = {}
    metadata = {
        "finger_face_count": {},
        "distal_face_count": {},
        "distal_start_fraction": {},
    }
    for finger in FINGER_NAMES:
        finger_vertices = vertex_finger == finger
        distal_vertices = finger_vertices & (vertex_path_fraction >= max(0.0, distal_start_fraction[finger] - 0.05))
        finger_faces[finger] = _faces_from_vertex_mask(faces, finger_vertices)
        distal_faces[finger] = _faces_from_vertex_mask(faces, distal_vertices)
        metadata["finger_face_count"][finger] = int(finger_faces[finger].shape[0])
        metadata["distal_face_count"][finger] = int(distal_faces[finger].shape[0])
        metadata["distal_start_fraction"][finger] = float(distal_start_fraction[finger])

    return ManoSemanticRegions(
        finger_faces=finger_faces,
        distal_faces=distal_faces,
        vertex_finger=vertex_finger,
        vertex_path_fraction=vertex_path_fraction,
        metadata=metadata,
    )


def normalize_finger_name(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.lower().replace("-", "_")
    for token in cleaned.split("_"):
        token = FINGER_ALIASES.get(token, token)
        if token in FINGER_NAMES:
            return token
    return None


def infer_finger_from_name(value: str | None, *, fallback: str | None = None) -> str | None:
    finger = normalize_finger_name(value)
    if finger is not None:
        return finger
    return normalize_finger_name(fallback)


def _project_points_semantic_distal(
    points: np.ndarray,
    layout: TactileLayout,
    mesh: Mesh,
    regions: ManoSemanticRegions,
    global_faces: np.ndarray,
    global_barycentric: np.ndarray,
    global_projected: np.ndarray,
    global_distances: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str | None, ...],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    nearest_faces = np.empty(points.shape[0], dtype=np.int64)
    barycentric = np.empty((points.shape[0], 3), dtype=np.float64)
    projected = np.empty_like(points)
    distances = np.empty(points.shape[0], dtype=np.float64)
    target_fingers: list[str | None] = []
    candidate_face_counts = np.zeros(points.shape[0], dtype=np.int64)
    fallback_reasons: list[str] = []
    warning_mask = np.zeros(points.shape[0], dtype=np.bool_)
    semantic_target_points = points.copy()
    normalized_coordinates = _empty_normalized_coordinates(points.shape[0])

    for idx, point in enumerate(points):
        finger = infer_finger_from_name(layout.semantic_regions[idx], fallback=layout.sensor_names[idx])
        target_fingers.append(finger)
        if finger is None:
            _copy_global_projection(
                idx,
                global_faces,
                global_barycentric,
                global_projected,
                global_distances,
                nearest_faces,
                barycentric,
                projected,
                distances,
            )
            candidate_face_counts[idx] = int(mesh.faces.shape[0])
            fallback_reasons.append("unknown_finger_to_global")
            continue

        candidate_faces = regions.distal_faces.get(finger, np.zeros(0, dtype=np.int64))
        fallback = "none"
        if candidate_faces.size == 0:
            candidate_faces = regions.finger_faces.get(finger, np.zeros(0, dtype=np.int64))
            fallback = "empty_distal_to_finger"
        if candidate_faces.size == 0:
            _copy_global_projection(
                idx,
                global_faces,
                global_barycentric,
                global_projected,
                global_distances,
                nearest_faces,
                barycentric,
                projected,
                distances,
            )
            candidate_face_counts[idx] = int(mesh.faces.shape[0])
            fallback_reasons.append("empty_finger_to_global")
            continue

        face, bary, proj, distance = _project_single_point_to_faces(point, mesh, candidate_faces)
        nearest_faces[idx] = face
        barycentric[idx] = bary
        projected[idx] = proj
        distances[idx] = distance
        candidate_face_counts[idx] = int(candidate_faces.shape[0])
        fallback_reasons.append(fallback)
        warning_mask[idx] = (
            distance > global_distances[idx] + SEMANTIC_DISTANCE_WARN_DELTA_M
            or distance > SEMANTIC_DISTANCE_WARN_ABS_M
        )

    return (
        nearest_faces,
        barycentric,
        projected,
        distances,
        tuple(target_fingers),
        candidate_face_counts,
        tuple(fallback_reasons),
        warning_mask,
        semantic_target_points,
        normalized_coordinates,
    )


def _project_points_semantic_normalized(
    points_robot: np.ndarray,
    points_mano: np.ndarray,
    layout: TactileLayout,
    mesh: Mesh,
    regions: ManoSemanticRegions,
    robot_keypoints: Mapping[str, np.ndarray],
    mano_keypoints: Mapping[str, np.ndarray],
    distal_result: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[str | None, ...],
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    global_distances: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str | None, ...],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    (
        nearest_faces,
        barycentric,
        projected,
        distances,
        target_fingers,
        candidate_face_counts,
        fallback_reasons,
        warning_mask,
        semantic_target_points,
        normalized_coordinates,
    ) = (
        np.array(distal_result[0], copy=True),
        np.array(distal_result[1], copy=True),
        np.array(distal_result[2], copy=True),
        np.array(distal_result[3], copy=True),
        distal_result[4],
        np.array(distal_result[5], copy=True),
        list(distal_result[6]),
        np.array(distal_result[7], copy=True),
        np.array(distal_result[8], copy=True),
        np.array(distal_result[9], copy=True),
    )

    for idx, finger in enumerate(target_fingers):
        if finger is None:
            fallback_reasons[idx] = "normalized_unknown_finger_to_semantic_distal"
            continue
        candidate_faces = regions.distal_faces.get(finger, np.zeros(0, dtype=np.int64))
        if candidate_faces.size == 0:
            candidate_faces = regions.finger_faces.get(finger, np.zeros(0, dtype=np.int64))
        if candidate_faces.size == 0:
            fallback_reasons[idx] = "normalized_empty_candidates_to_semantic_distal"
            continue
        try:
            coordinate = _finger_local_coordinate(points_robot[idx], robot_keypoints, finger)
            target_point = _finger_point_from_coordinate(coordinate, mano_keypoints, finger)
        except ValueError:
            fallback_reasons[idx] = "normalized_degenerate_frame_to_semantic_distal"
            continue

        face, bary, proj, _target_distance = _project_single_point_to_faces(target_point, mesh, candidate_faces)
        nearest_faces[idx] = face
        barycentric[idx] = bary
        projected[idx] = proj
        distances[idx] = float(np.linalg.norm(proj - points_mano[idx]))
        candidate_face_counts[idx] = int(candidate_faces.shape[0])
        fallback_reasons[idx] = "none"
        semantic_target_points[idx] = target_point
        normalized_coordinates[idx] = np.asarray(
            [
                coordinate["path_fraction"],
                coordinate["segment_index"],
                coordinate["segment_fraction"],
                coordinate["v_norm"],
                coordinate["w_norm"],
            ],
            dtype=np.float64,
        )
        warning_mask[idx] = (
            distances[idx] > global_distances[idx] + SEMANTIC_DISTANCE_WARN_DELTA_M
            or distances[idx] > SEMANTIC_DISTANCE_WARN_ABS_M
        )

    return (
        nearest_faces,
        barycentric,
        projected,
        distances,
        target_fingers,
        candidate_face_counts,
        tuple(fallback_reasons),
        warning_mask,
        semantic_target_points,
        normalized_coordinates,
    )


def _project_points_block_preserving(
    points_robot: np.ndarray,
    points_mano: np.ndarray,
    layout: TactileLayout,
    mesh: Mesh,
    regions: ManoSemanticRegions,
    robot_to_mano: np.ndarray,
    robot_keypoints: Mapping[str, np.ndarray],
    mano_keypoints: Mapping[str, np.ndarray],
    normalized_result: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[str | None, ...],
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    global_distances: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str | None, ...],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
]:
    (
        nearest_faces,
        barycentric,
        projected,
        distances,
        target_fingers,
        candidate_face_counts,
        fallback_reasons,
        warning_mask,
        semantic_target_points,
        normalized_coordinates,
    ) = (
        np.array(normalized_result[0], copy=True),
        np.array(normalized_result[1], copy=True),
        np.array(normalized_result[2], copy=True),
        np.array(normalized_result[3], copy=True),
        normalized_result[4],
        np.array(normalized_result[5], copy=True),
        list(normalized_result[6]),
        np.array(normalized_result[7], copy=True),
        np.array(normalized_result[8], copy=True),
        np.array(normalized_result[9], copy=True),
    )

    block_ids = list(layout.sensor_names)
    block_local_uv = _empty_block_local_uv(points_mano.shape[0])
    block_target_points = np.array(semantic_target_points, copy=True)
    block_similarity_scale = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    block_fallback_reasons = ["not_applicable" for _ in range(points_mano.shape[0])]
    block_layout_preservation = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    robot_keypoints_mano = _transform_keypoints(robot_keypoints, robot_to_mano)

    for item in layout.sensor_slices:
        start = int(item["start"])
        count = int(item["count"])
        end = start + count
        indices = np.arange(start, end, dtype=np.int64)
        if indices.size == 0:
            continue

        block_name = str(item.get("name", ""))
        for idx in indices:
            block_ids[int(idx)] = block_name or layout.sensor_names[int(idx)]

        finger = infer_finger_from_name(str(item.get("semantic_region", "")), fallback=block_name)
        if finger is None:
            _mark_block_fallback(block_fallback_reasons, indices, "block_unknown_finger_to_semantic_normalized")
            continue
        if indices.size < 3:
            _mark_block_fallback(block_fallback_reasons, indices, "block_too_small_to_semantic_normalized")
            continue

        candidate_faces = regions.distal_faces.get(finger, np.zeros(0, dtype=np.int64))
        if candidate_faces.size == 0:
            candidate_faces = regions.finger_faces.get(finger, np.zeros(0, dtype=np.int64))
        if candidate_faces.size == 0:
            _mark_block_fallback(block_fallback_reasons, indices, "block_empty_candidates_to_semantic_normalized")
            continue

        try:
            uv, _source_u_axis, _source_v_axis = _block_local_coordinates(points_mano[indices], robot_keypoints_mano, finger)
            center_coordinate = _finger_local_coordinate(points_robot[indices].mean(axis=0), robot_keypoints, finger)
            target_center = _finger_point_from_coordinate(center_coordinate, mano_keypoints, finger)
            target_u_axis, target_v_axis, _target_normal_axis = _finger_frame_from_coordinate(
                center_coordinate,
                mano_keypoints,
                finger,
            )
            scale = _finger_distal_length_scale(robot_keypoints_mano, mano_keypoints, finger)
        except (KeyError, ValueError):
            _mark_block_fallback(block_fallback_reasons, indices, "block_degenerate_frame_to_semantic_normalized")
            continue

        target_points = target_center[None, :] + scale * (
            uv[:, 0:1] * target_u_axis[None, :] + uv[:, 1:2] * target_v_axis[None, :]
        )
        for local_idx, idx in enumerate(indices):
            face, bary, proj, _target_distance = _project_single_point_to_faces(
                target_points[local_idx],
                mesh,
                candidate_faces,
            )
            nearest_faces[idx] = face
            barycentric[idx] = bary
            projected[idx] = proj
            distances[idx] = float(np.linalg.norm(proj - points_mano[idx]))
            candidate_face_counts[idx] = int(candidate_faces.shape[0])
            fallback_reasons[idx] = "none"
            block_fallback_reasons[idx] = "none"
            semantic_target_points[idx] = target_points[local_idx]
            block_target_points[idx] = target_points[local_idx]
            block_local_uv[idx] = uv[local_idx]
            block_similarity_scale[idx] = scale
            warning_mask[idx] = (
                distances[idx] > global_distances[idx] + SEMANTIC_DISTANCE_WARN_DELTA_M
                or distances[idx] > SEMANTIC_DISTANCE_WARN_ABS_M
            )

        layout_error = _block_layout_preservation_error(uv, projected[indices], scale)
        block_layout_preservation[indices] = layout_error

    return (
        nearest_faces,
        barycentric,
        projected,
        distances,
        target_fingers,
        candidate_face_counts,
        tuple(fallback_reasons),
        warning_mask,
        semantic_target_points,
        normalized_coordinates,
        tuple(block_ids),
        block_local_uv,
        block_target_points,
        block_similarity_scale,
        tuple(block_fallback_reasons),
        block_layout_preservation,
    )


def _project_points_graph_preserving(
    layout: TactileLayout,
    points_mano: np.ndarray,
    mesh: Mesh,
    regions: ManoSemanticRegions,
    block_result: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[str | None, ...],
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
    ],
    global_distances: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str | None, ...],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    (
        nearest_faces,
        barycentric,
        projected,
        distances,
        target_fingers,
        candidate_face_counts,
        fallback_reasons,
        warning_mask,
        semantic_target_points,
        normalized_coordinates,
        block_ids,
        block_local_uv,
        block_target_points,
        block_similarity_scale,
        block_fallback_reasons,
        block_layout_preservation,
    ) = (
        np.array(block_result[0], copy=True),
        np.array(block_result[1], copy=True),
        np.array(block_result[2], copy=True),
        np.array(block_result[3], copy=True),
        block_result[4],
        np.array(block_result[5], copy=True),
        list(block_result[6]),
        np.array(block_result[7], copy=True),
        np.array(block_result[8], copy=True),
        np.array(block_result[9], copy=True),
        block_result[10],
        np.array(block_result[11], copy=True),
        np.array(block_result[12], copy=True),
        np.array(block_result[13], copy=True),
        block_result[14],
        np.array(block_result[15], copy=True),
    )

    graph_knn_indices = _empty_graph_knn_indices(points_mano.shape[0])
    graph_edge_length_error = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    graph_neighbor_mismatch = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    graph_angle_error_rad = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    graph_laplacian_error = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    graph_fallback_reasons = ["not_applicable" for _ in range(points_mano.shape[0])]

    for item in layout.sensor_slices:
        start = int(item["start"])
        count = int(item["count"])
        end = start + count
        indices = np.arange(start, end, dtype=np.int64)
        if indices.size == 0:
            continue

        block_reason_set = set(block_fallback_reasons[start:end])
        if block_reason_set != {"none"}:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_block_fallback_to_block_preserving")
            continue

        finger = infer_finger_from_name(str(item.get("semantic_region", "")), fallback=str(item.get("name", "")))
        if finger is None:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_unknown_finger_to_block_preserving")
            continue
        if indices.size < 3:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_too_small_to_block_preserving")
            continue

        candidate_faces = regions.distal_faces.get(finger, np.zeros(0, dtype=np.int64))
        if candidate_faces.size == 0:
            candidate_faces = regions.finger_faces.get(finger, np.zeros(0, dtype=np.int64))
        if candidate_faces.size == 0:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_empty_candidates_to_block_preserving")
            continue

        source_points = block_local_uv[indices] * block_similarity_scale[indices, None]
        anchor_points = block_target_points[indices]
        initial_points = projected[indices]
        if not (
            np.isfinite(source_points).all()
            and np.isfinite(anchor_points).all()
            and np.isfinite(initial_points).all()
        ):
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_missing_block_coordinates_to_block_preserving")
            continue

        k = min(GRAPH_PRESERVING_K, int(indices.size) - 1)
        if k < 2:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_too_small_to_block_preserving")
            continue
        knn_local = _knn_indices(source_points, k)
        graph_knn_indices[indices, :k] = indices[knn_local]

        initial_metrics = _graph_preservation_metrics(source_points, initial_points, anchor_points, knn_local)
        refined_faces, refined_barycentric, refined_points = _refine_graph_points_on_faces(
            source_points,
            initial_points,
            anchor_points,
            mesh,
            candidate_faces,
            knn_local,
        )
        final_metrics = _graph_preservation_metrics(source_points, refined_points, anchor_points, knn_local)
        if _graph_preservation_score(final_metrics) <= _graph_preservation_score(initial_metrics) * 1.02 + 1e-9:
            metrics = final_metrics
            for local_idx, idx in enumerate(indices):
                nearest_faces[idx] = refined_faces[local_idx]
                barycentric[idx] = refined_barycentric[local_idx]
                projected[idx] = refined_points[local_idx]
                distances[idx] = float(np.linalg.norm(refined_points[local_idx] - points_mano[idx]))
                fallback_reasons[idx] = "none"
                graph_fallback_reasons[idx] = "none"
                warning_mask[idx] = (
                    distances[idx] > global_distances[idx] + SEMANTIC_DISTANCE_WARN_DELTA_M
                    or distances[idx] > SEMANTIC_DISTANCE_WARN_ABS_M
                )
        else:
            metrics = initial_metrics
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_refinement_rejected_to_block_preserving")

        graph_edge_length_error[indices] = metrics["edge_length_error"]
        graph_neighbor_mismatch[indices] = metrics["neighbor_mismatch"]
        graph_angle_error_rad[indices] = metrics["angle_error_rad"]
        graph_laplacian_error[indices] = metrics["laplacian_error"]

    return (
        nearest_faces,
        barycentric,
        projected,
        distances,
        target_fingers,
        candidate_face_counts,
        tuple(fallback_reasons),
        warning_mask,
        semantic_target_points,
        normalized_coordinates,
        block_ids,
        block_local_uv,
        block_target_points,
        block_similarity_scale,
        block_fallback_reasons,
        block_layout_preservation,
        graph_knn_indices,
        graph_edge_length_error,
        graph_neighbor_mismatch,
        graph_angle_error_rad,
        graph_laplacian_error,
        tuple(graph_fallback_reasons),
    )


def _project_points_graph_preserving_no_block(
    points_mano: np.ndarray,
    layout: TactileLayout,
    mesh: Mesh,
    regions: ManoSemanticRegions,
    robot_to_mano: np.ndarray,
    robot_keypoints: Mapping[str, np.ndarray],
    normalized_result: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[str | None, ...],
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    global_distances: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str | None, ...],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    (
        nearest_faces,
        barycentric,
        projected,
        distances,
        target_fingers,
        candidate_face_counts,
        fallback_reasons,
        warning_mask,
        semantic_target_points,
        normalized_coordinates,
    ) = (
        np.array(normalized_result[0], copy=True),
        np.array(normalized_result[1], copy=True),
        np.array(normalized_result[2], copy=True),
        np.array(normalized_result[3], copy=True),
        normalized_result[4],
        np.array(normalized_result[5], copy=True),
        list(normalized_result[6]),
        np.array(normalized_result[7], copy=True),
        np.array(normalized_result[8], copy=True),
        np.array(normalized_result[9], copy=True),
    )

    block_ids = list(layout.sensor_names)
    block_local_uv = _empty_block_local_uv(points_mano.shape[0])
    block_target_points = np.array(semantic_target_points, copy=True)
    block_similarity_scale = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    block_fallback_reasons = ["block_preserving_disabled" for _ in range(points_mano.shape[0])]
    block_layout_preservation = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    graph_knn_indices = _empty_graph_knn_indices(points_mano.shape[0])
    graph_edge_length_error = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    graph_neighbor_mismatch = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    graph_angle_error_rad = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    graph_laplacian_error = np.full(points_mano.shape[0], np.nan, dtype=np.float64)
    graph_fallback_reasons = ["not_applicable" for _ in range(points_mano.shape[0])]
    robot_keypoints_mano = _transform_keypoints(robot_keypoints, robot_to_mano)

    for item in layout.sensor_slices:
        start = int(item["start"])
        count = int(item["count"])
        end = start + count
        indices = np.arange(start, end, dtype=np.int64)
        if indices.size == 0:
            continue

        block_name = str(item.get("name", ""))
        for idx in indices:
            block_ids[int(idx)] = block_name or layout.sensor_names[int(idx)]

        if set(fallback_reasons[start:end]) != {"none"}:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_semantic_fallback_to_semantic_normalized")
            continue

        finger = infer_finger_from_name(str(item.get("semantic_region", "")), fallback=block_name)
        if finger is None:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_unknown_finger_to_semantic_normalized")
            continue
        if indices.size < 3:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_too_small_to_semantic_normalized")
            continue

        candidate_faces = regions.distal_faces.get(finger, np.zeros(0, dtype=np.int64))
        if candidate_faces.size == 0:
            candidate_faces = regions.finger_faces.get(finger, np.zeros(0, dtype=np.int64))
        if candidate_faces.size == 0:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_empty_candidates_to_semantic_normalized")
            continue

        try:
            source_points, _source_u_axis, _source_v_axis = _block_local_coordinates(
                points_mano[indices],
                robot_keypoints_mano,
                finger,
            )
        except (KeyError, ValueError):
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_degenerate_source_to_semantic_normalized")
            continue

        anchor_points = semantic_target_points[indices]
        initial_points = projected[indices]
        if not (
            np.isfinite(source_points).all()
            and np.isfinite(anchor_points).all()
            and np.isfinite(initial_points).all()
        ):
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_missing_source_to_semantic_normalized")
            continue

        k = min(GRAPH_PRESERVING_K, int(indices.size) - 1)
        if k < 2:
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_too_small_to_semantic_normalized")
            continue
        knn_local = _knn_indices(source_points, k)
        graph_knn_indices[indices, :k] = indices[knn_local]

        initial_metrics = _graph_preservation_metrics(source_points, initial_points, anchor_points, knn_local)
        refined_faces, refined_barycentric, refined_points = _refine_graph_points_on_faces(
            source_points,
            initial_points,
            anchor_points,
            mesh,
            candidate_faces,
            knn_local,
        )
        final_metrics = _graph_preservation_metrics(source_points, refined_points, anchor_points, knn_local)
        if _graph_preservation_score(final_metrics) <= _graph_preservation_score(initial_metrics) * 1.02 + 1e-9:
            metrics = final_metrics
            for local_idx, idx in enumerate(indices):
                nearest_faces[idx] = refined_faces[local_idx]
                barycentric[idx] = refined_barycentric[local_idx]
                projected[idx] = refined_points[local_idx]
                distances[idx] = float(np.linalg.norm(refined_points[local_idx] - points_mano[idx]))
                fallback_reasons[idx] = "none"
                graph_fallback_reasons[idx] = "none"
                block_local_uv[idx] = source_points[local_idx]
                block_similarity_scale[idx] = 1.0
                warning_mask[idx] = (
                    distances[idx] > global_distances[idx] + SEMANTIC_DISTANCE_WARN_DELTA_M
                    or distances[idx] > SEMANTIC_DISTANCE_WARN_ABS_M
                )
        else:
            metrics = initial_metrics
            _mark_block_fallback(graph_fallback_reasons, indices, "graph_refinement_rejected_to_semantic_normalized")
            block_local_uv[indices] = source_points
            block_similarity_scale[indices] = 1.0

        graph_edge_length_error[indices] = metrics["edge_length_error"]
        graph_neighbor_mismatch[indices] = metrics["neighbor_mismatch"]
        graph_angle_error_rad[indices] = metrics["angle_error_rad"]
        graph_laplacian_error[indices] = metrics["laplacian_error"]

    return (
        nearest_faces,
        barycentric,
        projected,
        distances,
        target_fingers,
        candidate_face_counts,
        tuple(fallback_reasons),
        warning_mask,
        semantic_target_points,
        normalized_coordinates,
        tuple(block_ids),
        block_local_uv,
        block_target_points,
        block_similarity_scale,
        tuple(block_fallback_reasons),
        block_layout_preservation,
        graph_knn_indices,
        graph_edge_length_error,
        graph_neighbor_mismatch,
        graph_angle_error_rad,
        graph_laplacian_error,
        tuple(graph_fallback_reasons),
    )


def _project_single_point_to_faces(
    point: np.ndarray,
    mesh: Mesh,
    candidate_face_indices: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    face_indices, barycentric, projected, distances = project_points_to_mesh(
        np.asarray(point, dtype=np.float64)[None, :],
        mesh,
        candidate_face_indices=candidate_face_indices,
    )
    return int(face_indices[0]), barycentric[0], projected[0], float(distances[0])


def _finger_local_coordinate(point: np.ndarray, keypoints: Mapping[str, np.ndarray], finger: str) -> dict[str, float]:
    joints = np.asarray([keypoints[label] for label in FINGER_KEYPOINT_LABELS[finger]], dtype=np.float64)
    wrist = np.asarray(keypoints.get("wrist", joints[0]), dtype=np.float64)
    lengths = np.linalg.norm(joints[1:] - joints[:-1], axis=1)
    total_length = float(lengths.sum())
    if total_length <= 1e-18:
        raise ValueError(f"{finger} keypoints are degenerate")

    best: dict[str, float] | None = None
    best_dist2 = np.inf
    path_offset = 0.0
    point = np.asarray(point, dtype=np.float64)
    for segment_index, (start, end, length) in enumerate(zip(joints[:-1], joints[1:], lengths)):
        if length <= 1e-18:
            path_offset += float(length)
            continue
        direction = end - start
        segment_fraction = float(np.clip(((point - start) @ direction) / (length * length), 0.0, 1.0))
        closest = start + segment_fraction * direction
        tangent, side_axis, normal_axis = _finger_segment_frame(start, end, wrist)
        offset = point - closest
        v_norm = float((offset @ side_axis) / length)
        w_norm = float((offset @ normal_axis) / length)
        dist2 = float(np.dot(offset, offset))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best = {
                "path_fraction": float((path_offset + segment_fraction * length) / total_length),
                "segment_index": float(segment_index),
                "segment_fraction": segment_fraction,
                "v_norm": v_norm,
                "w_norm": w_norm,
            }
        path_offset += float(length)

    if best is None:
        raise ValueError(f"{finger} keypoints do not contain a usable segment")
    return best


def _finger_point_from_coordinate(
    coordinate: Mapping[str, float],
    keypoints: Mapping[str, np.ndarray],
    finger: str,
) -> np.ndarray:
    joints = np.asarray([keypoints[label] for label in FINGER_KEYPOINT_LABELS[finger]], dtype=np.float64)
    wrist = np.asarray(keypoints.get("wrist", joints[0]), dtype=np.float64)
    lengths = np.linalg.norm(joints[1:] - joints[:-1], axis=1)
    total_length = float(lengths.sum())
    if total_length <= 1e-18:
        raise ValueError(f"{finger} keypoints are degenerate")

    path_target = float(np.clip(coordinate["path_fraction"], 0.0, 1.0)) * total_length
    path_offset = 0.0
    for start, end, length in zip(joints[:-1], joints[1:], lengths):
        if length <= 1e-18:
            continue
        if path_target <= path_offset + length or np.isclose(path_offset + length, total_length):
            segment_fraction = float(np.clip((path_target - path_offset) / length, 0.0, 1.0))
            tangent, side_axis, normal_axis = _finger_segment_frame(start, end, wrist)
            base = start + segment_fraction * (end - start)
            return (
                base
                + float(coordinate["v_norm"]) * length * side_axis
                + float(coordinate["w_norm"]) * length * normal_axis
            )
        path_offset += float(length)
    return joints[-1]


def _finger_frame_from_coordinate(
    coordinate: Mapping[str, float],
    keypoints: Mapping[str, np.ndarray],
    finger: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = np.asarray([keypoints[label] for label in FINGER_KEYPOINT_LABELS[finger]], dtype=np.float64)
    wrist = np.asarray(keypoints.get("wrist", joints[0]), dtype=np.float64)
    lengths = np.linalg.norm(joints[1:] - joints[:-1], axis=1)
    total_length = float(lengths.sum())
    if total_length <= 1e-18:
        raise ValueError(f"{finger} keypoints are degenerate")

    path_target = float(np.clip(coordinate["path_fraction"], 0.0, 1.0)) * total_length
    path_offset = 0.0
    for start, end, length in zip(joints[:-1], joints[1:], lengths):
        if length <= 1e-18:
            continue
        if path_target <= path_offset + length or np.isclose(path_offset + length, total_length):
            return _finger_segment_frame(start, end, wrist)
        path_offset += float(length)
    return _finger_segment_frame(joints[-2], joints[-1], wrist)


def _finger_segment_frame(
    start: np.ndarray,
    end: np.ndarray,
    wrist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tangent = _normalize_vector(end - start)
    reference = ((start + end) * 0.5) - wrist
    side_axis = reference - tangent * float(reference @ tangent)
    if np.linalg.norm(side_axis) <= 1e-9:
        side_axis = _fallback_perpendicular(tangent)
    else:
        side_axis = _normalize_vector(side_axis)
    normal_axis = _normalize_vector(np.cross(tangent, side_axis))
    side_axis = _normalize_vector(np.cross(normal_axis, tangent))
    return tangent, side_axis, normal_axis


def _block_local_coordinates(
    block_points: np.ndarray,
    robot_keypoints_mano: Mapping[str, np.ndarray],
    finger: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_points = np.asarray(block_points, dtype=np.float64)
    centered = block_points - block_points.mean(axis=0, keepdims=True)
    if centered.shape[0] < 3:
        raise ValueError("block must contain at least three points")
    try:
        _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("block PCA failed") from exc
    if singular_values.shape[0] < 2 or float(singular_values[1]) <= BLOCK_PRESERVING_PCA_MIN_SINGULAR:
        raise ValueError("block PCA plane is degenerate")

    plane_normal = _normalize_vector(np.cross(vt[0], vt[1]))
    dip_label = FINGER_KEYPOINT_LABELS[finger][2]
    tip_label = FINGER_KEYPOINT_LABELS[finger][3]
    finger_axis = _normalize_vector(robot_keypoints_mano[tip_label] - robot_keypoints_mano[dip_label])
    source_u_axis = finger_axis - plane_normal * float(finger_axis @ plane_normal)
    if np.linalg.norm(source_u_axis) <= 1e-9:
        source_u_axis = vt[0]
        if float(source_u_axis @ finger_axis) < 0.0:
            source_u_axis = -source_u_axis
    else:
        source_u_axis = _normalize_vector(source_u_axis)

    source_v_axis = _normalize_vector(np.cross(plane_normal, source_u_axis))
    if float(source_v_axis @ vt[1]) < 0.0:
        source_v_axis = -source_v_axis
    uv = np.stack([centered @ source_u_axis, centered @ source_v_axis], axis=1)
    return uv, source_u_axis, source_v_axis


def _finger_distal_length_scale(
    robot_keypoints_mano: Mapping[str, np.ndarray],
    mano_keypoints: Mapping[str, np.ndarray],
    finger: str,
) -> float:
    dip_label = FINGER_KEYPOINT_LABELS[finger][2]
    tip_label = FINGER_KEYPOINT_LABELS[finger][3]
    robot_length = float(np.linalg.norm(robot_keypoints_mano[tip_label] - robot_keypoints_mano[dip_label]))
    mano_length = float(np.linalg.norm(mano_keypoints[tip_label] - mano_keypoints[dip_label]))
    if robot_length <= 1e-18 or mano_length <= 1e-18:
        raise ValueError(f"{finger} distal length is degenerate")
    return float(np.clip(mano_length / robot_length, *BLOCK_PRESERVING_SCALE_RANGE))


def _block_layout_preservation_error(uv: np.ndarray, projected: np.ndarray, scale: float) -> float:
    if uv.shape[0] < 2:
        return float("nan")
    expected_points = float(scale) * np.asarray(uv, dtype=np.float64)
    actual_points = np.asarray(projected, dtype=np.float64)
    pair_i, pair_j = np.triu_indices(expected_points.shape[0], k=1)
    expected = np.linalg.norm(expected_points[pair_i] - expected_points[pair_j], axis=1)
    actual = np.linalg.norm(actual_points[pair_i] - actual_points[pair_j], axis=1)
    valid = expected > 1e-12
    if not np.any(valid):
        return float("nan")
    relative_error = np.abs(actual[valid] - expected[valid]) / expected[valid]
    return float(relative_error.mean())


def _refine_graph_points_on_faces(
    source_points: np.ndarray,
    initial_points: np.ndarray,
    anchor_points: np.ndarray,
    mesh: Mesh,
    candidate_faces: np.ndarray,
    knn_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(initial_points, dtype=np.float64).copy()
    anchors = np.asarray(anchor_points, dtype=np.float64)
    edges = _unique_knn_edges(knn_indices)
    desired_edge_lengths = {
        edge: float(np.linalg.norm(source_points[edge[0]] - source_points[edge[1]])) for edge in edges
    }

    for _iteration in range(GRAPH_PRESERVING_ITERATIONS):
        gradients = GRAPH_PRESERVING_DATA_WEIGHT * (points - anchors)
        for i, j in edges:
            diff = points[i] - points[j]
            length = float(np.linalg.norm(diff))
            desired = desired_edge_lengths[(i, j)]
            if length <= 1e-12 or desired <= 1e-12:
                continue
            force = (length - desired) * diff / length
            gradients[i] += GRAPH_PRESERVING_EDGE_WEIGHT * force
            gradients[j] -= GRAPH_PRESERVING_EDGE_WEIGHT * force

        anchor_laplacian = _graph_laplacian(anchors, knn_indices)
        current_laplacian = _graph_laplacian(points, knn_indices)
        gradients += GRAPH_PRESERVING_LAPLACIAN_WEIGHT * (current_laplacian - anchor_laplacian)
        points = points - GRAPH_PRESERVING_STEP_SIZE * gradients
        _faces, _barycentric, points = _snap_points_to_faces(points, mesh, candidate_faces)

    return _snap_points_to_faces(points, mesh, candidate_faces)


def _graph_preservation_metrics(
    source_points: np.ndarray,
    target_points: np.ndarray,
    anchor_points: np.ndarray,
    knn_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    anchor_points = np.asarray(anchor_points, dtype=np.float64)
    count = source_points.shape[0]
    edge_error = np.full(count, np.nan, dtype=np.float64)
    neighbor_mismatch = np.full(count, np.nan, dtype=np.float64)
    angle_error = np.full(count, np.nan, dtype=np.float64)
    laplacian_error = np.full(count, np.nan, dtype=np.float64)
    target_knn = _knn_indices(target_points, knn_indices.shape[1])
    anchor_laplacian = _graph_laplacian(anchor_points, knn_indices)
    target_laplacian = _graph_laplacian(target_points, knn_indices)

    for idx in range(count):
        neighbors = knn_indices[idx]
        source_set = {int(item) for item in neighbors if int(item) != idx}
        target_set = {int(item) for item in target_knn[idx] if int(item) != idx}
        if source_set:
            neighbor_mismatch[idx] = 1.0 - len(source_set & target_set) / float(len(source_set))

        local_edge_errors = []
        local_angle_errors = []
        for neighbor in neighbors:
            neighbor = int(neighbor)
            desired = float(np.linalg.norm(source_points[idx] - source_points[neighbor]))
            actual = float(np.linalg.norm(target_points[idx] - target_points[neighbor]))
            if desired > 1e-12:
                local_edge_errors.append(abs(actual - desired) / desired)
        for left_pos in range(len(neighbors)):
            for right_pos in range(left_pos + 1, len(neighbors)):
                left = int(neighbors[left_pos])
                right = int(neighbors[right_pos])
                source_angle = _vector_angle(
                    source_points[left] - source_points[idx],
                    source_points[right] - source_points[idx],
                )
                target_angle = _vector_angle(
                    target_points[left] - target_points[idx],
                    target_points[right] - target_points[idx],
                )
                if np.isfinite(source_angle) and np.isfinite(target_angle):
                    local_angle_errors.append(abs(target_angle - source_angle))
        if local_edge_errors:
            edge_error[idx] = float(np.mean(local_edge_errors))
        if local_angle_errors:
            angle_error[idx] = float(np.mean(local_angle_errors))

        scale = _mean_neighbor_distance(source_points, idx, neighbors)
        if scale > 1e-12:
            laplacian_error[idx] = float(np.linalg.norm(target_laplacian[idx] - anchor_laplacian[idx]) / scale)

    return {
        "edge_length_error": edge_error,
        "neighbor_mismatch": neighbor_mismatch,
        "angle_error_rad": angle_error,
        "laplacian_error": laplacian_error,
    }


def _graph_preservation_score(metrics: Mapping[str, np.ndarray]) -> float:
    edge = _finite_mean(metrics["edge_length_error"])
    neighbor = _finite_mean(metrics["neighbor_mismatch"])
    angle = _finite_mean(metrics["angle_error_rad"]) / np.pi
    laplacian = _finite_mean(metrics["laplacian_error"])
    return float(edge + 0.50 * neighbor + 0.25 * angle + 0.25 * laplacian)


def _knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"points must have shape (N, D), got {points.shape}")
    count = points.shape[0]
    if count < 2:
        return np.zeros((count, 0), dtype=np.int64)
    k = int(np.clip(k, 1, count - 1))
    diff = points[:, None, :] - points[None, :, :]
    distances = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1)[:, :k].astype(np.int64)


def _unique_knn_edges(knn_indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for idx, row in enumerate(knn_indices):
        for neighbor in row:
            neighbor = int(neighbor)
            if neighbor == idx:
                continue
            edges.add((min(idx, neighbor), max(idx, neighbor)))
    return tuple(sorted(edges))


def _graph_laplacian(points: np.ndarray, knn_indices: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    laplacian = np.zeros_like(points)
    for idx, neighbors in enumerate(knn_indices):
        if neighbors.size == 0:
            continue
        laplacian[idx] = points[idx] - points[neighbors].mean(axis=0)
    return laplacian


def _mean_neighbor_distance(points: np.ndarray, idx: int, neighbors: np.ndarray) -> float:
    if neighbors.size == 0:
        return 0.0
    distances = np.linalg.norm(points[neighbors] - points[idx][None, :], axis=1)
    finite = distances[np.isfinite(distances) & (distances > 1e-12)]
    if finite.size == 0:
        return 0.0
    return float(finite.mean())


def _vector_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
    return float(np.arccos(cosine))


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(finite.mean())


def _snap_points_to_faces(
    points: np.ndarray,
    mesh: Mesh,
    candidate_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faces = np.empty(points.shape[0], dtype=np.int64)
    barycentric = np.empty((points.shape[0], 3), dtype=np.float64)
    projected = np.empty_like(points)
    for idx, point in enumerate(points):
        face, bary, proj, _distance = _project_single_point_to_faces(point, mesh, candidate_faces)
        faces[idx] = face
        barycentric[idx] = bary
        projected[idx] = proj
    return faces, barycentric, projected


def _transform_keypoints(keypoints: Mapping[str, np.ndarray], transform: np.ndarray) -> dict[str, np.ndarray]:
    labels = list(keypoints.keys())
    points = np.asarray([keypoints[label] for label in labels], dtype=np.float64)
    transformed = apply_similarity_to_points(np.asarray(transform, dtype=np.float64), points)
    return {label: transformed[idx] for idx, label in enumerate(labels)}


def _mark_block_fallback(reasons: list[str], indices: np.ndarray, reason: str) -> None:
    for idx in indices:
        reasons[int(idx)] = reason


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-18:
        raise ValueError("cannot normalize a degenerate vector")
    return np.asarray(vector, dtype=np.float64) / norm


def _fallback_perpendicular(tangent: np.ndarray) -> np.ndarray:
    axes = np.eye(3, dtype=np.float64)
    axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
    return _normalize_vector(axis - tangent * float(axis @ tangent))


def _empty_normalized_coordinates(count: int) -> np.ndarray:
    return np.full((count, 5), np.nan, dtype=np.float64)


def _empty_block_local_uv(count: int) -> np.ndarray:
    return np.full((count, 2), np.nan, dtype=np.float64)


def _empty_graph_knn_indices(count: int) -> np.ndarray:
    return np.full((count, GRAPH_PRESERVING_K), -1, dtype=np.int64)


def closest_points_on_triangles(point: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(point, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {point.shape}")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"triangles must have shape (T, 3, 3), got {triangles.shape}")

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = point[None, :] - a

    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point[None, :] - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point[None, :] - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    bary = np.zeros((triangles.shape[0], 3), dtype=np.float64)

    mask_a = (d1 <= 0.0) & (d2 <= 0.0)
    bary[mask_a, 0] = 1.0

    mask_b = (d3 >= 0.0) & (d4 <= d3) & ~mask_a
    bary[mask_b, 1] = 1.0

    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0) & ~(mask_a | mask_b)
    denom_ab = d1 - d3
    v_ab = np.divide(d1, denom_ab, out=np.zeros_like(d1), where=np.abs(denom_ab) > 1e-18)
    bary[mask_ab, 0] = 1.0 - v_ab[mask_ab]
    bary[mask_ab, 1] = v_ab[mask_ab]

    mask_c = (d6 >= 0.0) & (d5 <= d6) & ~(mask_a | mask_b | mask_ab)
    bary[mask_c, 2] = 1.0

    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0) & ~(mask_a | mask_b | mask_ab | mask_c)
    denom_ac = d2 - d6
    w_ac = np.divide(d2, denom_ac, out=np.zeros_like(d2), where=np.abs(denom_ac) > 1e-18)
    bary[mask_ac, 0] = 1.0 - w_ac[mask_ac]
    bary[mask_ac, 2] = w_ac[mask_ac]

    va = d3 * d6 - d5 * d4
    assigned = mask_a | mask_b | mask_ab | mask_c | mask_ac
    mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0) & ~assigned
    denom_bc = (d4 - d3) + (d5 - d6)
    w_bc = np.divide(d4 - d3, denom_bc, out=np.zeros_like(d4), where=np.abs(denom_bc) > 1e-18)
    bary[mask_bc, 1] = 1.0 - w_bc[mask_bc]
    bary[mask_bc, 2] = w_bc[mask_bc]

    mask_face = ~(assigned | mask_bc)
    denom_face = va + vb + vc
    v = np.divide(vb, denom_face, out=np.zeros_like(vb), where=np.abs(denom_face) > 1e-18)
    w = np.divide(vc, denom_face, out=np.zeros_like(vc), where=np.abs(denom_face) > 1e-18)
    bary[mask_face, 0] = 1.0 - v[mask_face] - w[mask_face]
    bary[mask_face, 1] = v[mask_face]
    bary[mask_face, 2] = w[mask_face]

    degenerate = np.isclose(np.linalg.norm(np.cross(ab, ac), axis=1), 0.0)
    if np.any(degenerate):
        bary[degenerate] = _closest_degenerate_barycentric(point, triangles[degenerate])

    bary = np.asarray([_normalize_barycentric(row) for row in bary], dtype=np.float64)
    closest = (
        bary[:, 0:1] * triangles[:, 0]
        + bary[:, 1:2] * triangles[:, 1]
        + bary[:, 2:3] * triangles[:, 2]
    )
    return closest, bary


def load_obj_mesh(path: str | Path) -> Mesh:
    path = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                indices = [_parse_obj_index(token, len(vertices)) for token in parts[1:]]
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])
    if not vertices:
        raise ValueError(f"{path} does not contain OBJ vertices")
    if not faces:
        raise ValueError(f"{path} does not contain OBJ faces")
    return Mesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64))


def load_alignment_side_config(
    alignment_json: str | Path,
    side: str,
    *,
    fitted_mano_obj: str | Path | None = None,
) -> AlignmentSideConfig:
    side = _validate_side(side)
    alignment_path = Path(alignment_json)
    with alignment_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    side_payload = payload.get("sides", {}).get(side)
    if side_payload is None:
        raise ValueError(f"{alignment_path} does not contain side {side!r}")

    robot_to_mano = _extract_robot_to_mano(side_payload)
    qpos = _extract_robot_qpos(side_payload)
    mano_keypoints = _extract_mano_keypoints(side_payload)
    obj_path = Path(fitted_mano_obj) if fitted_mano_obj is not None else _extract_fitted_obj_path(payload, side)
    return AlignmentSideConfig(
        side=side,
        robot_to_mano=robot_to_mano,
        robot_qpos=qpos,
        mano_keypoints=mano_keypoints,
        fitted_mano_obj=obj_path,
        metadata={
            "alignment_json": str(alignment_path),
            "alignment_command": payload.get("command"),
            "fitted_mano_obj": str(obj_path) if obj_path is not None else None,
        },
    )


def save_projection_npz(result: ProjectionResult, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rows=result.rows.astype(np.int64),
        cols=result.cols.astype(np.int64),
        vals=result.vals.astype(np.float64),
        shape=np.asarray(result.shape, dtype=np.int64),
        nearest_face_indices=result.nearest_face_indices.astype(np.int64),
        barycentric=result.barycentric.astype(np.float64),
        global_nearest_face_indices=result.global_nearest_face_indices.astype(np.int64),
        global_barycentric=result.global_barycentric.astype(np.float64),
        tactile_points_robot=result.layout.points_robot.astype(np.float64),
        tactile_points_mano=result.tactile_points_mano.astype(np.float64),
        projected_points=result.projected_points.astype(np.float64),
        global_projected_points=result.global_projected_points.astype(np.float64),
        semantic_target_points=result.semantic_target_points.astype(np.float64),
        normalized_coordinates=result.normalized_coordinates.astype(np.float64),
        block_ids=np.asarray(result.block_ids),
        block_local_uv=result.block_local_uv.astype(np.float64),
        block_target_points=result.block_target_points.astype(np.float64),
        block_similarity_scale=result.block_similarity_scale.astype(np.float64),
        block_fallback_reasons=np.asarray(result.block_fallback_reasons),
        block_layout_preservation=result.block_layout_preservation.astype(np.float64),
        graph_knn_indices=result.graph_knn_indices.astype(np.int64),
        graph_edge_length_error=result.graph_edge_length_error.astype(np.float64),
        graph_neighbor_mismatch=result.graph_neighbor_mismatch.astype(np.float64),
        graph_angle_error_rad=result.graph_angle_error_rad.astype(np.float64),
        graph_laplacian_error=result.graph_laplacian_error.astype(np.float64),
        graph_fallback_reasons=np.asarray(result.graph_fallback_reasons),
        nearest_distances_m=result.distances_m.astype(np.float64),
        global_nearest_distances_m=result.global_distances_m.astype(np.float64),
        candidate_face_counts=result.candidate_face_counts.astype(np.int64),
        fallback_reasons=np.asarray(result.fallback_reasons),
        semantic_warning_mask=result.semantic_warning_mask.astype(np.bool_),
        target_fingers=np.asarray([finger or "" for finger in result.target_fingers]),
        vertex_weight_sum=result.vertex_weight_sum.astype(np.float64),
        valid_vertex_mask=result.valid_vertex_mask.astype(np.bool_),
        sensor_names=np.asarray(result.layout.sensor_names),
        semantic_regions=np.asarray(result.layout.semantic_regions),
        taxel_indices=result.layout.taxel_indices.astype(np.int64),
        robot_to_mano=result.robot_to_mano.astype(np.float64),
        projection_mode=np.asarray(result.projection_mode),
    )


def apply_sensor_values_to_vertices(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    shape: tuple[int, int],
    sensor_values: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    values = np.asarray(sensor_values, dtype=np.float64)
    if values.shape != (shape[1],):
        raise ValueError(f"sensor_values must have shape ({shape[1]},), got {values.shape}")
    vertex_values = np.zeros(shape[0], dtype=np.float64)
    np.add.at(vertex_values, np.asarray(rows, dtype=np.int64), np.asarray(vals, dtype=np.float64) * values[np.asarray(cols, dtype=np.int64)])
    if normalize:
        normalizer = np.zeros(shape[0], dtype=np.float64)
        np.add.at(normalizer, np.asarray(rows, dtype=np.int64), np.asarray(vals, dtype=np.float64))
        valid = normalizer > 0.0
        vertex_values[valid] /= normalizer[valid]
    return vertex_values


def projection_result_to_json(result: ProjectionResult) -> dict[str, Any]:
    column_sums = np.zeros(result.shape[1], dtype=np.float64)
    np.add.at(column_sums, result.cols, result.vals)
    distance_delta_m = result.distances_m - result.global_distances_m
    quality_evaluation = _quality_evaluation(result)
    return {
        "side": result.side,
        "projection_mode": result.projection_mode,
        "taxel_count": int(result.shape[1]),
        "mano_vertex_count": int(result.shape[0]),
        "mano_face_count": int(result.mano_mesh.faces.shape[0]),
        "valid_vertex_count": int(result.valid_vertex_mask.sum()),
        "nearest_distance_mm": {
            "mean": float(result.distances_m.mean() * 1000.0) if result.distances_m.size else 0.0,
            "max": float(result.distances_m.max() * 1000.0) if result.distances_m.size else 0.0,
            "rms": float(np.sqrt(np.mean(result.distances_m**2)) * 1000.0) if result.distances_m.size else 0.0,
        },
        "global_nearest_distance_mm": {
            "mean": float(result.global_distances_m.mean() * 1000.0) if result.global_distances_m.size else 0.0,
            "max": float(result.global_distances_m.max() * 1000.0) if result.global_distances_m.size else 0.0,
            "rms": float(np.sqrt(np.mean(result.global_distances_m**2)) * 1000.0)
            if result.global_distances_m.size
            else 0.0,
        },
        "semantic_distance_delta_mm": {
            "mean": float(distance_delta_m.mean() * 1000.0) if distance_delta_m.size else 0.0,
            "max": float(distance_delta_m.max() * 1000.0) if distance_delta_m.size else 0.0,
        },
        "semantic_projection": {
            "target_finger_counts": _count_strings(tuple(finger or "unknown" for finger in result.target_fingers)),
            "fallback_counts": _count_strings(result.fallback_reasons),
            "warning_count": int(result.semantic_warning_mask.sum()),
            "normalized_coordinate_finite_count": int(np.isfinite(result.normalized_coordinates[:, 0]).sum())
            if result.normalized_coordinates.size
            else 0,
            "candidate_face_count": {
                "min": int(result.candidate_face_counts.min()) if result.candidate_face_counts.size else 0,
                "max": int(result.candidate_face_counts.max()) if result.candidate_face_counts.size else 0,
            },
            "warning_thresholds": {
                "semantic_distance_warn_delta_mm": SEMANTIC_DISTANCE_WARN_DELTA_M * 1000.0,
                "semantic_distance_warn_abs_mm": SEMANTIC_DISTANCE_WARN_ABS_M * 1000.0,
            },
        },
        "block_projection": {
            "block_count": int(len(result.layout.sensor_slices)),
            "fallback_counts": _count_strings(result.block_fallback_reasons),
            "finite_uv_count": int(np.isfinite(result.block_local_uv).all(axis=1).sum())
            if result.block_local_uv.size
            else 0,
            "similarity_scale": _finite_stats(result.block_similarity_scale),
            "block_layout_preservation": {
                "metric": "mean_relative_pairwise_distance_error_after_surface_snap",
                **_finite_stats(result.block_layout_preservation),
            },
            "scale_clip_range": {
                "min": BLOCK_PRESERVING_SCALE_RANGE[0],
                "max": BLOCK_PRESERVING_SCALE_RANGE[1],
            },
        },
        "graph_projection": {
            "knn_k": GRAPH_PRESERVING_K,
            "iterations": GRAPH_PRESERVING_ITERATIONS,
            "step_size": GRAPH_PRESERVING_STEP_SIZE,
            "weights": {
                "data": GRAPH_PRESERVING_DATA_WEIGHT,
                "edge": GRAPH_PRESERVING_EDGE_WEIGHT,
                "laplacian": GRAPH_PRESERVING_LAPLACIAN_WEIGHT,
            },
            "fallback_counts": _count_strings(result.graph_fallback_reasons),
            "finite_knn_count": int((result.graph_knn_indices[:, 0] >= 0).sum())
            if result.graph_knn_indices.size
            else 0,
            "edge_length_error": {
                "metric": "mean_relative_knn_edge_length_error",
                **_finite_stats(result.graph_edge_length_error),
            },
            "neighbor_mismatch": {
                "metric": "one_minus_knn_neighbor_overlap",
                **_finite_stats(result.graph_neighbor_mismatch),
            },
            "angle_error_rad": {
                "metric": "mean_absolute_local_knn_angle_error_rad",
                **_finite_stats(result.graph_angle_error_rad),
            },
            "laplacian_error": {
                "metric": "normalized_knn_laplacian_error",
                **_finite_stats(result.graph_laplacian_error),
            },
        },
        "weight_column_sum": {
            "min": float(column_sums.min()) if column_sums.size else 0.0,
            "max": float(column_sums.max()) if column_sums.size else 0.0,
        },
        "quality_evaluation": quality_evaluation,
        "sensor_summaries": _sensor_summaries(result),
        "layout": {
            "metadata": _jsonable(result.layout.metadata),
            "sensor_slices": _jsonable(result.layout.sensor_slices),
        },
        "metadata": _jsonable(result.metadata),
    }


def render_projection_result(
    result: ProjectionResult,
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
    bounds_points = np.concatenate(
        [result.mano_mesh.vertices, result.tactile_points_mano, result.projected_points],
        axis=0,
    )
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
        draw.polygon(polygon, fill=(shade, shade, shade, 58), outline=(100, 100, 100, 28))

    tactile_xy, _ = to_screen(result.tactile_points_mano)
    nearest_xy, _ = to_screen(result.projected_points)
    palette = _sensor_palette()
    sensor_color = {item["name"]: palette[idx % len(palette)] for idx, item in enumerate(result.layout.sensor_slices)}
    for idx, (taxel_xy, mesh_xy) in enumerate(zip(tactile_xy, nearest_xy)):
        if idx % 3 == 0:
            draw.line(
                [(float(taxel_xy[0]), float(taxel_xy[1])), (float(mesh_xy[0]), float(mesh_xy[1]))],
                fill=(70, 70, 70, 45),
                width=1,
            )
    for idx, xy in enumerate(tactile_xy):
        color = sensor_color.get(result.layout.sensor_names[idx], (225, 75, 45))
        radius = 2
        draw.ellipse(
            (float(xy[0] - radius), float(xy[1] - radius), float(xy[0] + radius), float(xy[1] + radius)),
            fill=color + (230,),
            outline=(255, 255, 255, 180),
        )
        if _taxel_needs_warning_outline(result, idx):
            warn_radius = 4
            draw.ellipse(
                (
                    float(xy[0] - warn_radius),
                    float(xy[1] - warn_radius),
                    float(xy[0] + warn_radius),
                    float(xy[1] + warn_radius),
                ),
                outline=(20, 20, 20, 210),
                width=1,
            )
    for xy in nearest_xy[:: max(1, len(nearest_xy) // 240)]:
        radius = 1
        draw.ellipse(
            (float(xy[0] - radius), float(xy[1] - radius), float(xy[0] + radius), float(xy[1] + radius)),
            fill=(20, 20, 20, 150),
        )

    distance_mm = result.distances_m * 1000.0
    draw.text(
        (14, 12),
        title or f"{result.side} tactile layout -> fitted MANO mesh | {result.projection_mode}",
        fill=(20, 20, 20, 255),
    )
    fallback_count = _projection_fallback_count(result)
    draw.text(
        (14, height - 30),
        f"taxels={result.shape[1]} vertices={result.shape[0]} valid_vertices={int(result.valid_vertex_mask.sum())} "
        f"mean/max nearest={distance_mm.mean():.2f}/{distance_mm.max():.2f} mm "
        f"warnings={int(result.semantic_warning_mask.sum())} fallbacks={fallback_count}",
        fill=(75, 75, 75, 255),
    )
    image.convert("RGB").save(out_path)


def render_projected_surface_result(
    result: ProjectionResult,
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
    bounds_points = np.concatenate([result.mano_mesh.vertices, result.projected_points], axis=0)
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
        draw.polygon(polygon, fill=(shade, shade, shade, 64), outline=(92, 92, 92, 32))

    projected_xy, _ = to_screen(result.projected_points)
    palette = _sensor_palette()
    sensor_color = {item["name"]: palette[idx % len(palette)] for idx, item in enumerate(result.layout.sensor_slices)}
    for idx, xy in enumerate(projected_xy):
        color = sensor_color.get(result.layout.sensor_names[idx], (225, 75, 45))
        radius = 2
        draw.ellipse(
            (float(xy[0] - radius), float(xy[1] - radius), float(xy[0] + radius), float(xy[1] + radius)),
            fill=color + (235,),
            outline=(255, 255, 255, 190),
        )
        if _taxel_needs_warning_outline(result, idx):
            warn_radius = 4
            draw.ellipse(
                (
                    float(xy[0] - warn_radius),
                    float(xy[1] - warn_radius),
                    float(xy[0] + warn_radius),
                    float(xy[1] + warn_radius),
                ),
                outline=(20, 20, 20, 215),
                width=1,
            )

    distance_mm = result.distances_m * 1000.0
    draw.text(
        (14, 12),
        title or f"{result.side} projected tactile surface on MANO | {result.projection_mode}",
        fill=(20, 20, 20, 255),
    )
    fallback_count = _projection_fallback_count(result)
    draw.text(
        (14, height - 30),
        f"surface taxels={result.shape[1]} valid_vertices={int(result.valid_vertex_mask.sum())} "
        f"mean/max nearest={distance_mm.mean():.2f}/{distance_mm.max():.2f} mm "
        f"warnings={int(result.semantic_warning_mask.sum())} fallbacks={fallback_count}",
        fill=(75, 75, 75, 255),
    )
    image.convert("RGB").save(out_path)


def _taxel_needs_warning_outline(result: ProjectionResult, idx: int) -> bool:
    if bool(result.semantic_warning_mask[idx]):
        return True
    if result.fallback_reasons[idx].endswith("_global"):
        return True
    if result.block_fallback_reasons[idx] not in {"none", "not_applicable", "block_preserving_disabled"}:
        return True
    return result.graph_fallback_reasons[idx] not in {"none", "not_applicable"}


def _projection_fallback_count(result: ProjectionResult) -> int:
    count = 0
    for idx in range(result.shape[1]):
        has_semantic_fallback = result.fallback_reasons[idx] not in {"none", "not_applicable"}
        has_block_fallback = result.block_fallback_reasons[idx] not in {
            "none",
            "not_applicable",
            "block_preserving_disabled",
        }
        has_graph_fallback = result.graph_fallback_reasons[idx] not in {"none", "not_applicable"}
        if has_semantic_fallback or has_block_fallback or has_graph_fallback:
            count += 1
    return count


def _sensor_summaries(result: ProjectionResult) -> list[dict[str, Any]]:
    summaries = []
    for item in result.layout.sensor_slices:
        start = int(item["start"])
        end = start + int(item["count"])
        distances = result.distances_m[start:end] * 1000.0
        global_distances = result.global_distances_m[start:end] * 1000.0
        warning_count = int(result.semantic_warning_mask[start:end].sum())
        fallback_counts = _count_strings(result.fallback_reasons[start:end])
        block_fallback_counts = _count_strings(result.block_fallback_reasons[start:end])
        graph_fallback_counts = _count_strings(result.graph_fallback_reasons[start:end])
        target_fingers = sorted({finger for finger in result.target_fingers[start:end] if finger is not None})
        candidate_counts = result.candidate_face_counts[start:end]
        summaries.append(
            {
                "name": item["name"],
                "semantic_region": item["semantic_region"],
                "link_name": item["link_name"],
                "taxel_count": int(item["count"]),
                "target_fingers": target_fingers,
                "candidate_face_count": {
                    "min": int(candidate_counts.min()) if candidate_counts.size else 0,
                    "max": int(candidate_counts.max()) if candidate_counts.size else 0,
                },
                "fallback_counts": fallback_counts,
                "block_fallback_counts": block_fallback_counts,
                "graph_fallback_counts": graph_fallback_counts,
                "semantic_warning_count": warning_count,
                "block_similarity_scale": _finite_stats(result.block_similarity_scale[start:end]),
                "block_layout_preservation": {
                    "metric": "mean_relative_pairwise_distance_error_after_surface_snap",
                    **_finite_stats(result.block_layout_preservation[start:end]),
                },
                "graph_edge_length_error": _finite_stats(result.graph_edge_length_error[start:end]),
                "graph_neighbor_mismatch": _finite_stats(result.graph_neighbor_mismatch[start:end]),
                "graph_angle_error_rad": _finite_stats(result.graph_angle_error_rad[start:end]),
                "graph_laplacian_error": _finite_stats(result.graph_laplacian_error[start:end]),
                "quality_evaluation": _quality_evaluation(result, start=start, end=end),
                "nearest_distance_mm": {
                    "mean": float(distances.mean()) if distances.size else 0.0,
                    "max": float(distances.max()) if distances.size else 0.0,
                },
                "global_nearest_distance_mm": {
                    "mean": float(global_distances.mean()) if global_distances.size else 0.0,
                    "max": float(global_distances.max()) if global_distances.size else 0.0,
                },
                "semantic_distance_delta_mm": {
                    "mean": float((distances - global_distances).mean()) if distances.size else 0.0,
                    "max": float((distances - global_distances).max()) if distances.size else 0.0,
                },
            }
        )
    return summaries


def _quality_evaluation(
    result: ProjectionResult,
    *,
    start: int | None = None,
    end: int | None = None,
) -> dict[str, Any]:
    taxel_slice = slice(start, end)
    distances = result.distances_m[taxel_slice]
    global_distances = result.global_distances_m[taxel_slice]
    distance_delta = distances - global_distances
    warning_mask = result.semantic_warning_mask[taxel_slice]
    projected_points = result.projected_points[taxel_slice]
    nearest_faces = result.nearest_face_indices[taxel_slice]
    graph_edge = result.graph_edge_length_error[taxel_slice]
    graph_neighbor = result.graph_neighbor_mismatch[taxel_slice]
    graph_angle = result.graph_angle_error_rad[taxel_slice]
    graph_laplacian = result.graph_laplacian_error[taxel_slice]
    taxel_count = int(projected_points.shape[0])

    return {
        "surface_fitting": {
            "nearest_distance_mm": _finite_stats(distances * 1000.0),
            "nearest_distance_rms_mm": float(np.sqrt(np.mean(distances**2)) * 1000.0) if distances.size else 0.0,
            "global_nearest_distance_mm": _finite_stats(global_distances * 1000.0),
            "semantic_distance_delta_mm": _finite_stats(distance_delta * 1000.0),
            "warning_count": int(warning_mask.sum()),
            "warning_ratio": float(warning_mask.mean()) if warning_mask.size else 0.0,
        },
        "graph_preservation": {
            "edge_length_error": {
                "metric": "mean_relative_knn_edge_length_error",
                "preference": "lower_is_better",
                **_finite_stats(graph_edge),
            },
            "neighbor_mismatch": {
                "metric": "one_minus_knn_neighbor_overlap",
                "preference": "lower_is_better",
                **_finite_stats(graph_neighbor),
            },
            "angle_error_rad": {
                "metric": "mean_absolute_local_knn_angle_error_rad",
                "preference": "lower_is_better",
                **_finite_stats(graph_angle),
            },
            "laplacian_error": {
                "metric": "normalized_knn_laplacian_error",
                "preference": "lower_is_better",
                **_finite_stats(graph_laplacian),
            },
        },
        "distribution_quality": _distribution_quality_metrics(
            projected_points,
            nearest_faces,
            result.mano_mesh,
            taxel_count=taxel_count,
        ),
    }


def _distribution_quality_metrics(
    points: np.ndarray,
    nearest_faces: np.ndarray,
    mesh: Mesh,
    *,
    taxel_count: int,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    nearest_faces = np.asarray(nearest_faces, dtype=np.int64)
    nearest_neighbor_distances = _nearest_neighbor_distances(points)
    nearest_neighbor_mm = nearest_neighbor_distances * 1000.0
    coverage_area_m2 = _pca_coverage_area(points)
    occupied_faces = np.unique(nearest_faces).size if nearest_faces.size else 0
    occupied_vertices = (
        np.unique(np.asarray(mesh.faces, dtype=np.int64)[nearest_faces].reshape(-1)).size
        if nearest_faces.size
        else 0
    )
    return {
        "nearest_neighbor_distance_mm": {
            **_finite_stats(nearest_neighbor_mm),
            "cv": _coefficient_of_variation(nearest_neighbor_mm),
            "metric": "nearest_projected_taxel_spacing",
            "cv_preference": "lower_is_more_uniform",
        },
        "collapse_ratio": {
            "threshold_mm": COLLAPSE_DISTANCE_THRESHOLD_M * 1000.0,
            "value": _collapse_ratio(points, COLLAPSE_DISTANCE_THRESHOLD_M),
            "preference": "lower_is_better",
        },
        "coverage_area_mm2": {
            "metric": "convex_hull_area_in_first_two_pca_axes",
            "value": float(coverage_area_m2 * 1_000_000.0),
            "preference": "compare_per_sensor_not_monotonic",
        },
        "pca_spread_mm": _pca_spread_stats(points),
        "occupied_face_count": int(occupied_faces),
        "occupied_face_ratio": float(occupied_faces / max(1, mesh.faces.shape[0])),
        "occupied_vertex_count": int(occupied_vertices),
        "occupied_vertex_ratio": float(occupied_vertices / max(1, mesh.vertices.shape[0])),
        "taxel_count": int(taxel_count),
    }


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return np.zeros(points.shape[0], dtype=np.float64)
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _collapse_ratio(points: np.ndarray, threshold: float) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return 0.0
    pair_i, pair_j = np.triu_indices(points.shape[0], k=1)
    distances = np.linalg.norm(points[pair_i] - points[pair_j], axis=1)
    if distances.size == 0:
        return 0.0
    return float(np.mean(distances < threshold))


def _coefficient_of_variation(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    mean = float(finite.mean())
    if abs(mean) <= 1e-18:
        return None
    return float(finite.std() / mean)


def _pca_spread_stats(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return {"axis0": 0.0, "axis1": 0.0, "axis2": 0.0}
    centered = points - points.mean(axis=0, keepdims=True)
    if points.shape[0] == 1:
        return {"axis0": 0.0, "axis1": 0.0, "axis2": 0.0}
    _u, singular_values, _vt = np.linalg.svd(centered, full_matrices=False)
    spreads = np.zeros(3, dtype=np.float64)
    count_scale = np.sqrt(max(1, points.shape[0] - 1))
    spreads[: min(3, singular_values.size)] = singular_values[:3] / count_scale * 1000.0
    return {
        "axis0": float(spreads[0]),
        "axis1": float(spreads[1]),
        "axis2": float(spreads[2]),
    }


def _pca_coverage_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    if singular_values.size < 2 or float(singular_values[1]) <= 1e-18:
        return 0.0
    projected = centered @ vt[:2].T
    hull = _convex_hull_2d(projected)
    return _polygon_area_2d(hull)


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if points.shape[0] <= 1:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    sorted_points = points[order]

    def cross(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        return float((left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64)


def _polygon_area_2d(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _coerce_mano_keypoints(mano_keypoints: Mapping[str, Any] | np.ndarray) -> dict[str, np.ndarray]:
    if isinstance(mano_keypoints, Mapping):
        keypoints = {str(label): np.asarray(value, dtype=np.float64) for label, value in mano_keypoints.items()}
    else:
        labels = (
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
        points = np.asarray(mano_keypoints, dtype=np.float64)
        if points.shape != (len(labels), 3):
            raise ValueError(f"MANO keypoint array must have shape (21, 3), got {points.shape}")
        keypoints = {label: points[idx] for idx, label in enumerate(labels)}

    missing = sorted({label for labels in FINGER_KEYPOINT_LABELS.values() for label in labels} - set(keypoints))
    if missing:
        raise ValueError(f"MANO keypoints missing semantic labels: {missing}")
    for label, point in keypoints.items():
        if point.shape != (3,):
            raise ValueError(f"MANO keypoint {label!r} must have shape (3,), got {point.shape}")
    return keypoints


def _point_polyline_distances(points: np.ndarray, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    segments = joints[1:] - joints[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    total = float(lengths.sum())
    if total <= 1e-18:
        raise ValueError("finger keypoints are degenerate")

    best_dist2 = np.full(points.shape[0], np.inf, dtype=np.float64)
    best_path = np.zeros(points.shape[0], dtype=np.float64)
    offset = 0.0
    for start, direction, length in zip(joints[:-1], segments, lengths):
        if length <= 1e-18:
            continue
        rel = points - start[None, :]
        t = np.clip((rel @ direction) / (length * length), 0.0, 1.0)
        closest = start[None, :] + t[:, None] * direction[None, :]
        diff = points - closest
        dist2 = np.einsum("ij,ij->i", diff, diff)
        update = dist2 < best_dist2
        best_dist2[update] = dist2[update]
        best_path[update] = offset + t[update] * length
        offset += float(length)
    return np.sqrt(best_dist2), best_path / total


def _distal_start_fraction(joints: np.ndarray) -> float:
    lengths = np.linalg.norm(np.asarray(joints[1:] - joints[:-1], dtype=np.float64), axis=1)
    total = float(lengths.sum())
    if total <= 1e-18:
        return 0.65
    return float((lengths[0] + lengths[1]) / total)


def _faces_from_vertex_mask(faces: np.ndarray, vertex_mask: np.ndarray) -> np.ndarray:
    counts = vertex_mask[np.asarray(faces, dtype=np.int64)].sum(axis=1)
    selected = np.flatnonzero(counts >= 2).astype(np.int64)
    if selected.size == 0:
        selected = np.flatnonzero(counts >= 1).astype(np.int64)
    return selected


def _copy_global_projection(
    idx: int,
    global_faces: np.ndarray,
    global_barycentric: np.ndarray,
    global_projected: np.ndarray,
    global_distances: np.ndarray,
    nearest_faces: np.ndarray,
    barycentric: np.ndarray,
    projected: np.ndarray,
    distances: np.ndarray,
) -> None:
    nearest_faces[idx] = global_faces[idx]
    barycentric[idx] = global_barycentric[idx]
    projected[idx] = global_projected[idx]
    distances[idx] = global_distances[idx]


def _count_strings(values: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"finite_count": 0, "mean": None, "min": None, "max": None}
    return {
        "finite_count": int(finite.size),
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _extract_robot_to_mano(side_payload: Mapping[str, Any]) -> np.ndarray:
    candidates = [
        side_payload.get("fixed_frame_report", {}).get("sources", {}).get("robot", {}).get("robot_to_mano"),
        side_payload.get("sources", {}).get("robot", {}).get("robot_to_mano"),
        side_payload.get("transform", {}).get("robot_to_mano"),
        side_payload.get("fixed_frame_report", {}).get("transform", {}).get("robot_to_mano"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        matrix = np.asarray(candidate, dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix
    raise ValueError("alignment JSON side payload does not contain a valid robot_to_mano transform")


def _extract_robot_qpos(side_payload: Mapping[str, Any]) -> dict[str, float]:
    candidates = [
        side_payload.get("robot_qpos"),
        side_payload.get("fixed_frame_report", {}).get("sources", {}).get("robot", {}).get("qpos"),
        side_payload.get("reference_pose_fit", {}).get("qpos"),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return {str(name): float(value) for name, value in candidate.items()}
    return {}


def _extract_mano_keypoints(side_payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    candidates = [
        side_payload.get("fixed_frame_report", {}).get("keypoints"),
        side_payload.get("keypoints"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, list):
            continue
        keypoints: dict[str, np.ndarray] = {}
        for item in candidate:
            if not isinstance(item, Mapping) or "label" not in item or "mano" not in item:
                continue
            keypoints[str(item["label"])] = np.asarray(item["mano"], dtype=np.float64)
        if keypoints:
            return _coerce_mano_keypoints(keypoints)
    return {}


def _extract_fitted_obj_path(payload: Mapping[str, Any], side: str) -> Path | None:
    obj = payload.get("outputs", {}).get("obj")
    if isinstance(obj, Mapping) and obj.get(side):
        return Path(obj[side])
    if isinstance(obj, str):
        return Path(obj)
    return None


def _closest_degenerate_barycentric(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    bary = np.zeros((triangles.shape[0], 3), dtype=np.float64)
    distances = np.linalg.norm(triangles - point[None, None, :], axis=2)
    closest = np.argmin(distances, axis=1)
    bary[np.arange(triangles.shape[0]), closest] = 1.0
    return bary


def _normalize_barycentric(values: np.ndarray) -> np.ndarray:
    bary = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    total = float(bary.sum())
    if total <= 1e-18:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return bary / total


def _parse_obj_index(token: str, vertex_count: int) -> int:
    raw = token.split("/")[0]
    index = int(raw)
    if index > 0:
        return index - 1
    return vertex_count + index


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


def _sensor_palette() -> tuple[tuple[int, int, int], ...]:
    return (
        (225, 78, 54),
        (39, 118, 214),
        (45, 157, 98),
        (189, 98, 27),
        (132, 91, 196),
        (33, 151, 160),
    )


def _validate_side(side: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    return side


def _validate_projection_mode(mode: str) -> str:
    if mode not in PROJECTION_MODES:
        raise ValueError(f"projection_mode must be one of {PROJECTION_MODES}, got {mode!r}")
    return mode


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
