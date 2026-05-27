from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from tactile_layout_3d_projection.geometry import transform_points
from tactile_layout_3d_projection.inspire import (
    Inspire3DScene,
    SensorPointCloud,
    export_sensor_points_3d,
    infer_urdf_root_link,
    load_urdf_collision_meshes,
    render_inspire_3d_scene,
    render_inspire_3d_scenes,
)
from tactile_layout_3d_projection.urdf import load_urdf


FINGER_LINKS = {
    "thumb": "{side}_hand_thumb_rota_link2",
    "index": "{side}_hand_index_rota_link2",
    "mid": "{side}_hand_mid_link2",
    "ring": "{side}_hand_ring_link2",
    "pinky": "{side}_hand_pinky_link2",
}


def build_xhand_3d_scene(
    urdf_path: str | Path,
    tactile_dir: str | Path,
    side: str,
) -> Inspire3DScene:
    model = load_urdf(urdf_path)
    root_link = infer_urdf_root_link(model.links, model.joints)
    link_transforms = model.link_transforms(root_link=root_link)
    vertices, faces = load_urdf_collision_meshes(urdf_path, link_transforms)

    sensors = []
    tactile_dir = Path(tactile_dir)
    for finger, link_template in FINGER_LINKS.items():
        link_name = link_template.format(side=side)
        if link_name not in link_transforms:
            raise ValueError(f"{side} XHand URDF missing link {link_name!r}")
        mujoco_points = _load_tactile_xml_points(tactile_dir / f"{finger}_collisions.xml")
        local_points = _mujoco_tactile_points_to_urdf_local(mujoco_points, finger, side)
        world_points = transform_points(link_transforms[link_name], local_points)
        sensors.append(
            SensorPointCloud(
                name=f"{side}_{finger}_tactile_sensor",
                side=side,
                semantic_region=f"{side}_{finger}",
                grid_shape=(local_points.shape[0], 1),
                points=world_points,
                taxel_indices=np.stack(
                    [np.arange(local_points.shape[0], dtype=np.int64), np.zeros(local_points.shape[0], dtype=np.int64)],
                    axis=1,
                ),
            )
        )

    return Inspire3DScene(side=side, vertices=vertices, faces=faces, sensors=tuple(sensors))


def render_xhand_3d_scenes(
    scenes: list[Inspire3DScene],
    out_path: str | Path,
    view_mode: str = "same-camera",
) -> None:
    if len(scenes) == 1:
        render_inspire_3d_scene(scenes[0], out_path, view_mode=view_mode, title=f"{scenes[0].side} XHand")
    else:
        render_inspire_3d_scenes(scenes, out_path, view_mode=view_mode, title_suffix="XHand")


def export_xhand_sensor_points_3d(scenes: list[Inspire3DScene], out_path: str | Path) -> None:
    export_sensor_points_3d(scenes, out_path)


def _load_tactile_xml_points(path: Path) -> np.ndarray:
    root = ET.parse(path).getroot()
    points = []
    for geom in root.findall(".//geom"):
        if geom.attrib.get("class") != "pad":
            continue
        raw = geom.attrib.get("pos")
        if raw is None:
            continue
        values = [float(v) for v in raw.split()]
        if len(values) != 3:
            raise ValueError(f"{path} contains invalid geom pos {raw!r}")
        points.append(values)
    if not points:
        raise ValueError(f"{path} does not contain tactile pad geoms")
    return np.asarray(points, dtype=np.float64)


def _mujoco_tactile_points_to_urdf_local(points: np.ndarray, finger: str, side: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if finger == "thumb":
        side_sign = 1.0 if side == "right" else -1.0
        return np.stack([points[:, 1], side_sign * points[:, 0], -points[:, 2]], axis=1)

    return np.stack([points[:, 0], points[:, 1], -points[:, 2]], axis=1)
