from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from tactile_layout_3d_projection.geometry import transform_matrix


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass
class URDFModel:
    path: Path
    name: str
    links: set[str]
    joints: list[Joint]

    @property
    def parent_to_joints(self) -> dict[str, list[Joint]]:
        out: dict[str, list[Joint]] = {}
        for joint in self.joints:
            out.setdefault(joint.parent, []).append(joint)
        return out

    @property
    def child_to_joint(self) -> dict[str, Joint]:
        return {joint.child: joint for joint in self.joints}

    def validate_mesh_references(self) -> list[str]:
        root = ET.parse(self.path).getroot()
        missing: list[str] = []
        for mesh in root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if not filename:
                continue
            if filename.startswith("package://"):
                continue
            if not (self.path.parent / filename).exists():
                missing.append(filename)
        return missing

    def link_transforms(self, qpos: dict[str, float] | None = None, root_link: str = "base") -> dict[str, np.ndarray]:
        qpos = expand_mimic_joints(self.joints, qpos or {})
        transforms = {root_link: np.eye(4, dtype=np.float64)}
        pending = list(self.parent_to_joints.get(root_link, []))
        while pending:
            joint = pending.pop(0)
            if joint.parent not in transforms:
                pending.append(joint)
                continue
            joint_tf = transform_matrix(joint.origin_xyz, joint.origin_rpy)
            motion_tf = _joint_motion_transform(joint, qpos.get(joint.name, 0.0))
            transforms[joint.child] = transforms[joint.parent] @ joint_tf @ motion_tf
            pending.extend(self.parent_to_joints.get(joint.child, []))
        return transforms


def load_urdf(path: str | Path) -> URDFModel:
    path = Path(path)
    root = ET.parse(path).getroot()
    links = {link.attrib["name"] for link in root.findall("link")}
    joints = [_parse_joint(joint_el) for joint_el in root.findall("joint")]
    return URDFModel(path=path, name=root.attrib.get("name", path.stem), links=links, joints=joints)


def expand_mimic_joints(joints: list[Joint], qpos: dict[str, float]) -> dict[str, float]:
    expanded = {str(key): float(value) for key, value in qpos.items()}
    changed = True
    while changed:
        changed = False
        for joint in joints:
            if joint.mimic_joint and joint.mimic_joint in expanded and joint.name not in expanded:
                expanded[joint.name] = expanded[joint.mimic_joint] * joint.mimic_multiplier + joint.mimic_offset
                changed = True
    return expanded


def _parse_joint(joint_el: ET.Element) -> Joint:
    origin_el = joint_el.find("origin")
    origin_xyz = _parse_vector(origin_el.attrib.get("xyz", "0 0 0") if origin_el is not None else "0 0 0")
    origin_rpy = _parse_vector(origin_el.attrib.get("rpy", "0 0 0") if origin_el is not None else "0 0 0")
    axis_el = joint_el.find("axis")
    mimic_el = joint_el.find("mimic")
    return Joint(
        name=joint_el.attrib["name"],
        joint_type=joint_el.attrib.get("type", "fixed"),
        parent=joint_el.find("parent").attrib["link"],
        child=joint_el.find("child").attrib["link"],
        origin_xyz=origin_xyz,
        origin_rpy=origin_rpy,
        axis=_parse_vector(axis_el.attrib.get("xyz", "0 0 0")) if axis_el is not None else None,
        mimic_joint=mimic_el.attrib.get("joint") if mimic_el is not None else None,
        mimic_multiplier=float(mimic_el.attrib.get("multiplier", 1.0)) if mimic_el is not None else 1.0,
        mimic_offset=float(mimic_el.attrib.get("offset", 0.0)) if mimic_el is not None else 0.0,
    )


def _parse_vector(raw: str) -> tuple[float, float, float]:
    values = [float(v) for v in raw.split()]
    if len(values) != 3:
        raise ValueError(f"expected 3-vector, got {raw!r}")
    return values[0], values[1], values[2]


def _joint_motion_transform(joint: Joint, value: float) -> np.ndarray:
    if joint.joint_type == "fixed":
        return np.eye(4, dtype=np.float64)
    if joint.joint_type == "revolute":
        if joint.axis is None:
            raise ValueError(f"revolute joint {joint.name} has no axis")
        axis = np.asarray(joint.axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ValueError(f"revolute joint {joint.name} has zero axis")
        axis = axis / norm
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = _axis_angle_matrix(axis, float(value))
        return mat
    raise NotImplementedError(f"joint type {joint.joint_type!r} is not supported")


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )
