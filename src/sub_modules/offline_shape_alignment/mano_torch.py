from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sub_modules.offline_shape_alignment.mano import (
    _load_mano_pickle,
    _validate_side,
    default_mano_model_path,
    infer_mano_fingertip_vertices,
)
from sub_modules.offline_shape_alignment.types import KEYPOINT_LABELS, KeypointSet, Mesh


@dataclass(frozen=True)
class MANOShapeInstance:
    side: str
    beta: np.ndarray
    pose: np.ndarray | None
    mesh: Mesh
    joints: np.ndarray
    keypoints: KeypointSet
    fingertip_vertex_indices: dict[str, int]


@dataclass(frozen=True)
class MANOBetaModel:
    side: str
    model_path: Path
    num_betas: int
    faces: np.ndarray
    fingertip_vertex_indices: dict[str, int]
    v_template: Any
    shapedirs: Any
    j_regressor: Any
    posedirs: Any
    weights: Any
    parents: tuple[int, ...]

    def vertices_from_beta(self, beta: Any) -> Any:
        torch = _require_torch()
        beta = torch.as_tensor(beta, dtype=self.v_template.dtype, device=self.v_template.device)
        if beta.ndim != 1 or int(beta.shape[0]) != self.num_betas:
            raise ValueError(f"expected beta shape ({self.num_betas},), got {tuple(beta.shape)}")
        return self.v_template + torch.einsum("vci,i->vc", self.shapedirs, beta)

    def joints_from_vertices(self, vertices: Any) -> Any:
        torch = _require_torch()
        return torch.matmul(self.j_regressor, vertices)

    def keypoints_from_vertices(self, vertices: Any, joints: Any | None = None) -> Any:
        torch = _require_torch()
        if joints is None:
            joints = self.joints_from_vertices(vertices)
        points = {
            "wrist": joints[0],
            "thumb_mcp": joints[13],
            "thumb_pip": joints[14],
            "thumb_dip": joints[15],
            "thumb_tip": vertices[self.fingertip_vertex_indices["thumb"]],
            "index_mcp": joints[1],
            "index_pip": joints[2],
            "index_dip": joints[3],
            "index_tip": vertices[self.fingertip_vertex_indices["index"]],
            "middle_mcp": joints[4],
            "middle_pip": joints[5],
            "middle_dip": joints[6],
            "middle_tip": vertices[self.fingertip_vertex_indices["middle"]],
            "ring_mcp": joints[10],
            "ring_pip": joints[11],
            "ring_dip": joints[12],
            "ring_tip": vertices[self.fingertip_vertex_indices["ring"]],
            "pinky_mcp": joints[7],
            "pinky_pip": joints[8],
            "pinky_dip": joints[9],
            "pinky_tip": vertices[self.fingertip_vertex_indices["pinky"]],
        }
        return torch.stack([points[label] for label in KEYPOINT_LABELS], dim=0)

    def vertices_joints_from_beta_pose(self, beta: Any, pose: Any) -> tuple[Any, Any]:
        torch = _require_torch()
        beta = torch.as_tensor(beta, dtype=self.v_template.dtype, device=self.v_template.device)
        pose = torch.as_tensor(pose, dtype=self.v_template.dtype, device=self.v_template.device)
        if pose.shape == (15, 3):
            pose = torch.cat([torch.zeros((1, 3), dtype=pose.dtype, device=pose.device), pose], dim=0)
        if pose.shape != (16, 3):
            raise ValueError(f"expected pose shape (15, 3) or (16, 3), got {tuple(pose.shape)}")

        v_shaped = self.vertices_from_beta(beta)
        joints = self.joints_from_vertices(v_shaped)
        rot_mats = axis_angle_to_matrix(pose)
        eye = torch.eye(3, dtype=v_shaped.dtype, device=v_shaped.device)
        pose_feature = (rot_mats[1:] - eye[None, :, :]).reshape(-1)
        v_posed = v_shaped + torch.einsum("vcp,p->vc", self.posedirs, pose_feature)
        transforms = _global_rigid_transforms(rot_mats, joints, self.parents)
        joints_homo = torch.cat([joints, torch.zeros((joints.shape[0], 1), dtype=joints.dtype, device=joints.device)], dim=1)
        rest_offsets = torch.matmul(transforms, joints_homo[:, :, None])
        rel_transforms = transforms.clone()
        rel_transforms[:, :, 3:4] = rel_transforms[:, :, 3:4] - rest_offsets
        vertex_transforms = torch.einsum("vj,jab->vab", self.weights, rel_transforms)
        vertices_homo = torch.cat(
            [v_posed, torch.ones((v_posed.shape[0], 1), dtype=v_posed.dtype, device=v_posed.device)],
            dim=1,
        )
        vertices = torch.matmul(vertex_transforms, vertices_homo[:, :, None])[:, :3, 0]
        posed_joints = transforms[:, :3, 3]
        return vertices, posed_joints

    def instance(self, beta: np.ndarray | None = None, pose: np.ndarray | None = None) -> MANOShapeInstance:
        torch = _require_torch()
        if beta is None:
            beta = np.zeros(self.num_betas, dtype=np.float64)
        with torch.no_grad():
            beta_tensor = torch.as_tensor(beta, dtype=self.v_template.dtype, device=self.v_template.device)
            pose_tensor = None
            if pose is None:
                vertices = self.vertices_from_beta(beta_tensor)
                joints = self.joints_from_vertices(vertices)
            else:
                pose_tensor = torch.as_tensor(pose, dtype=self.v_template.dtype, device=self.v_template.device)
                vertices, joints = self.vertices_joints_from_beta_pose(beta_tensor, pose_tensor)
            keypoints = self.keypoints_from_vertices(vertices, joints)
        vertices_np = vertices.detach().cpu().numpy().astype(np.float64)
        joints_np = joints.detach().cpu().numpy().astype(np.float64)
        keypoints_np = keypoints.detach().cpu().numpy().astype(np.float64)
        beta_np = beta_tensor.detach().cpu().numpy().astype(np.float64)
        pose_np = None if pose_tensor is None else pose_tensor.detach().cpu().numpy().astype(np.float64)
        return MANOShapeInstance(
            side=self.side,
            beta=beta_np,
            pose=pose_np,
            mesh=Mesh(vertices=vertices_np, faces=self.faces.copy()),
            joints=joints_np,
            keypoints=KeypointSet(
                side=self.side,
                labels=KEYPOINT_LABELS,
                points=keypoints_np,
                metadata={
                    "hand": "mano",
                    "model_path": str(self.model_path),
                    "pose": "template_zero_pose" if pose_np is None else "optimized_pose_residual",
                    "beta": beta_np.tolist(),
                    "pose_residual": None if pose_np is None else pose_np.tolist(),
                    "fingertip_vertex_indices": dict(self.fingertip_vertex_indices),
                },
            ),
            fingertip_vertex_indices=dict(self.fingertip_vertex_indices),
        )


def load_mano_beta_model(
    side: str,
    mano_root: str | Path,
    *,
    num_betas: int = 10,
    device: str = "cpu",
    dtype: str = "float32",
) -> MANOBetaModel:
    torch = _require_torch()
    side = _validate_side(side)
    model_path = default_mano_model_path(mano_root, side)
    data = _load_mano_pickle(model_path)

    v_template = np.asarray(data["v_template"], dtype=np.float64)
    shapedirs = extract_shapedirs(data["shapedirs"]).astype(np.float64)
    if shapedirs.shape[:2] != v_template.shape:
        raise ValueError(f"MANO shapedirs shape {shapedirs.shape} does not match v_template {v_template.shape}")
    if num_betas <= 0 or num_betas > shapedirs.shape[2]:
        raise ValueError(f"num_betas must be in [1, {shapedirs.shape[2]}], got {num_betas}")

    faces = np.asarray(data["f"], dtype=np.int64)
    j_regressor = _dense_matrix(data["J_regressor"]).astype(np.float64)
    posedirs = np.asarray(data["posedirs"], dtype=np.float64)
    weights = np.asarray(data["weights"], dtype=np.float64)
    parents = _parents_from_kintree(np.asarray(data["kintree_table"], dtype=np.int64))
    fingertip_vertex_indices = infer_mano_fingertip_vertices(v_template, j_regressor @ v_template)
    torch_dtype = getattr(torch, dtype)

    return MANOBetaModel(
        side=side,
        model_path=model_path,
        num_betas=int(num_betas),
        faces=faces,
        fingertip_vertex_indices=fingertip_vertex_indices,
        v_template=torch.as_tensor(v_template, dtype=torch_dtype, device=device),
        shapedirs=torch.as_tensor(shapedirs[:, :, :num_betas], dtype=torch_dtype, device=device),
        j_regressor=torch.as_tensor(j_regressor, dtype=torch_dtype, device=device),
        posedirs=torch.as_tensor(posedirs, dtype=torch_dtype, device=device),
        weights=torch.as_tensor(weights, dtype=torch_dtype, device=device),
        parents=parents,
    )


def extract_shapedirs(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)

    preferred_shape = getattr(value, "preferred_shape", None)
    idxs = getattr(value, "idxs", None)
    source = getattr(value, "a", None)
    if source is None and hasattr(value, "__dict__"):
        source = value.__dict__.get("a")
    source_array = getattr(source, "x", None)
    if source_array is None and hasattr(source, "__dict__"):
        source_array = source.__dict__.get("x")
    if preferred_shape is not None and idxs is not None and source_array is not None:
        flat = np.asarray(source_array, dtype=np.float64).reshape(-1)
        return flat[np.asarray(idxs, dtype=np.int64)].reshape(tuple(preferred_shape))

    raise ValueError(f"unsupported MANO shapedirs object: {type(value)!r}")


def _dense_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _parents_from_kintree(kintree_table: np.ndarray) -> tuple[int, ...]:
    parents = []
    for idx, parent in enumerate(np.asarray(kintree_table, dtype=np.int64)[0]):
        parent_idx = int(parent)
        if idx == 0 or parent_idx < 0 or parent_idx > 1000000:
            parents.append(-1)
        else:
            parents.append(parent_idx)
    return tuple(parents)


def axis_angle_to_matrix(axis_angle: Any) -> Any:
    torch = _require_torch()
    axis_angle = torch.as_tensor(axis_angle)
    if axis_angle.ndim != 2 or axis_angle.shape[1] != 3:
        raise ValueError(f"expected axis_angle with shape Nx3, got {tuple(axis_angle.shape)}")
    theta2 = (axis_angle * axis_angle).sum(dim=1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    skew = _skew(axis_angle)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)[None, :, :]
    small = theta2 < 1e-8
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-16))
    return eye + a[:, :, None] * skew + b[:, :, None] * torch.matmul(skew, skew)


def _skew(vectors: Any) -> Any:
    torch = _require_torch()
    zero = torch.zeros(vectors.shape[0], dtype=vectors.dtype, device=vectors.device)
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return torch.stack(
        [
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ],
        dim=1,
    ).reshape(-1, 3, 3)


def _global_rigid_transforms(rot_mats: Any, joints: Any, parents: tuple[int, ...]) -> Any:
    torch = _require_torch()
    if len(parents) != int(joints.shape[0]):
        raise ValueError(f"expected {len(parents)} joints, got {joints.shape[0]}")

    rel_joints = joints.clone()
    for idx, parent in enumerate(parents):
        if parent >= 0:
            rel_joints[idx] = joints[idx] - joints[parent]

    transforms = []
    for idx, parent in enumerate(parents):
        local = _transform_matrix(rot_mats[idx], rel_joints[idx])
        if parent < 0:
            transforms.append(local)
        else:
            transforms.append(torch.matmul(transforms[parent], local))
    return torch.stack(transforms, dim=0)


def _transform_matrix(rotation: Any, translation: Any) -> Any:
    torch = _require_torch()
    transform = torch.eye(4, dtype=rotation.dtype, device=rotation.device)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


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
