from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from offline_shape_alignment.types import KEYPOINT_LABELS, KeypointSet, Mesh
from tactile_layout_3d_projection.inspire import infer_urdf_root_link, load_urdf_collision_meshes
from tactile_layout_3d_projection.urdf import URDFModel, load_urdf


DEFAULT_VIRTUAL_DIP_ALPHA = 0.65


@dataclass(frozen=True)
class XHandReference:
    side: str
    urdf_path: Path
    root_link: str
    mesh: Mesh
    keypoints: KeypointSet


def load_xhand_reference(
    urdf_path: str | Path,
    side: str,
    qpos: Mapping[str, float] | None = None,
) -> XHandReference:
    side = _validate_side(side)
    urdf_path = Path(urdf_path)
    model = load_urdf(urdf_path)
    root_link = infer_urdf_root_link(model.links, model.joints)
    link_transforms = model.link_transforms(dict(qpos or {}), root_link=root_link)
    vertices, faces = load_urdf_collision_meshes(urdf_path, link_transforms)
    keypoints = infer_xhand_semantic_keypoints(urdf_path, side, qpos=qpos)
    return XHandReference(
        side=side,
        urdf_path=urdf_path,
        root_link=root_link,
        mesh=Mesh(vertices=vertices, faces=faces),
        keypoints=keypoints,
    )


def infer_xhand_semantic_keypoints(
    urdf_path: str | Path,
    side: str,
    qpos: Mapping[str, float] | None = None,
) -> KeypointSet:
    side = _validate_side(side)
    model = load_urdf(urdf_path)
    root_link = infer_urdf_root_link(model.links, model.joints)
    link_transforms = model.link_transforms(dict(qpos or {}), root_link=root_link)
    joint_by_name = {joint.name: joint for joint in model.joints}

    points: dict[str, np.ndarray] = {"wrist": link_transforms[root_link][:3, 3].copy()}
    sources: dict[str, dict[str, object]] = {"wrist": {"type": "link_origin", "link": root_link}}

    def add_joint_point(label: str, joint_name: str) -> None:
        points[label] = _joint_origin_point(model, link_transforms, joint_by_name, joint_name)
        sources[label] = {"type": "joint_origin", "joint": joint_name}

    add_joint_point("thumb_mcp", f"{side}_hand_thumb_bend_joint")
    add_joint_point("thumb_pip", f"{side}_hand_thumb_rota_joint1")
    add_joint_point("thumb_dip", f"{side}_hand_thumb_rota_joint2")
    add_joint_point("thumb_tip", f"{side}_hand_thumb_rota_joint3")

    add_joint_point("index_mcp", f"{side}_hand_index_joint1")
    add_joint_point("index_pip", f"{side}_hand_index_joint2")
    add_joint_point("index_tip", f"{side}_hand_index_rota_joint3")
    points["index_dip"] = _virtual_point(points["index_pip"], points["index_tip"])
    sources["index_dip"] = _virtual_source("index_pip", "index_tip")

    for canonical, urdf_name in (("middle", "mid"), ("ring", "ring"), ("pinky", "pinky")):
        add_joint_point(f"{canonical}_mcp", f"{side}_hand_{urdf_name}_joint1")
        add_joint_point(f"{canonical}_pip", f"{side}_hand_{urdf_name}_joint2")
        add_joint_point(f"{canonical}_tip", f"{side}_hand_{urdf_name}_joint3")
        points[f"{canonical}_dip"] = _virtual_point(points[f"{canonical}_pip"], points[f"{canonical}_tip"])
        sources[f"{canonical}_dip"] = _virtual_source(f"{canonical}_pip", f"{canonical}_tip")

    ordered = np.stack([points[label] for label in KEYPOINT_LABELS], axis=0).astype(np.float64)
    return KeypointSet(
        side=side,
        labels=KEYPOINT_LABELS,
        points=ordered,
        metadata={
            "hand": "xhand",
            "urdf_path": str(Path(urdf_path)),
            "root_link": root_link,
            "qpos": {str(name): float(value) for name, value in sorted(dict(qpos or {}).items())},
            "virtual_dip_alpha": DEFAULT_VIRTUAL_DIP_ALPHA,
            "sources": sources,
        },
    )


def default_xhand_urdf_path(xhand_root: str | Path, side: str) -> Path:
    side = _validate_side(side)
    return Path(xhand_root) / "Xhand-urdf" / f"xhand1_{side}(1)" / "urdf" / f"xhand_{side}.urdf"


def _joint_origin_point(
    model: URDFModel,
    link_transforms: Mapping[str, np.ndarray],
    joint_by_name: Mapping[str, object],
    joint_name: str,
) -> np.ndarray:
    joint = joint_by_name.get(joint_name)
    if joint is None:
        raise ValueError(f"XHand URDF {model.path} is missing joint {joint_name!r}")
    child = getattr(joint, "child")
    transform = link_transforms.get(child)
    if transform is None:
        raise ValueError(f"XHand joint {joint_name!r} child link {child!r} is not reachable from root")
    return transform[:3, 3].copy()


def _virtual_point(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    return np.asarray(start, dtype=np.float64) + DEFAULT_VIRTUAL_DIP_ALPHA * (
        np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    )


def _virtual_source(start_label: str, end_label: str) -> dict[str, object]:
    return {
        "type": "virtual_between",
        "from": start_label,
        "to": end_label,
        "alpha": DEFAULT_VIRTUAL_DIP_ALPHA,
    }


def _validate_side(side: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    return side
