from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import struct
import xml.etree.ElementTree as ET

import numpy as np

from sub_modules.tactile_layout_3d_projection.config import SensorMountConfig
from sub_modules.tactile_layout_3d_projection.geometry import taxel_grid_points, transform_matrix, transform_points
from sub_modules.tactile_layout_3d_projection.urdf import load_urdf


@dataclass(frozen=True)
class SensorPointCloud:
    name: str
    side: str
    semantic_region: str
    grid_shape: tuple[int, int]
    points: np.ndarray
    taxel_indices: np.ndarray


@dataclass(frozen=True)
class Inspire3DScene:
    side: str
    vertices: np.ndarray
    faces: np.ndarray
    sensors: tuple[SensorPointCloud, ...]


def build_inspire_3d_scene(
    mount_config: SensorMountConfig,
    urdf_path: str | Path,
    side: str,
    qpos: dict[str, float] | None = None,
    root_link: str | None = "base",
) -> Inspire3DScene:
    model = load_urdf(urdf_path)
    if root_link is None or root_link not in model.links:
        root_link = infer_urdf_root_link(model.links, model.joints)
    link_transforms = model.link_transforms(qpos, root_link=root_link)
    vertices, faces = load_urdf_collision_meshes(urdf_path, link_transforms)

    sensors = []
    for sensor in mount_config.sensors:
        if sensor.side != side:
            continue
        if sensor.parent_link not in link_transforms:
            raise ValueError(f"sensor {sensor.name!r} parent_link {sensor.parent_link!r} not found in {side} URDF")
        local_points, taxel_indices = taxel_grid_points(sensor.size_mm, sensor.grid_shape)
        sensor_tf = transform_matrix(sensor.xyz, sensor.rpy)
        points = transform_points(link_transforms[sensor.parent_link] @ sensor_tf, local_points)
        sensors.append(
            SensorPointCloud(
                name=sensor.name,
                side=sensor.side,
                semantic_region=sensor.semantic_region,
                grid_shape=sensor.grid_shape,
                points=points,
                taxel_indices=taxel_indices,
            )
        )

    return Inspire3DScene(side=side, vertices=vertices, faces=faces, sensors=tuple(sensors))


def infer_urdf_root_link(links, joints) -> str:
    children = {joint.child for joint in joints}
    roots = sorted(set(links) - children)
    if not roots:
        raise ValueError("URDF has no root link")
    return roots[0]


def load_urdf_collision_meshes(
    urdf_path: str | Path,
    link_transforms: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    for link_el in root.findall("link"):
        link_name = link_el.attrib["name"]
        link_tf = link_transforms.get(link_name)
        if link_tf is None:
            continue
        for collision_el in link_el.findall("collision"):
            origin_tf = _origin_transform(collision_el.find("origin"))
            geom_el = collision_el.find("geometry")
            if geom_el is None:
                continue
            mesh = _geometry_mesh(geom_el, urdf_path.parent)
            if mesh is None:
                continue
            vertices, faces = mesh
            vertices = transform_points(link_tf @ origin_tf, vertices)
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += vertices.shape[0]

    if not all_vertices:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(all_vertices, axis=0), np.concatenate(all_faces, axis=0)


def render_inspire_3d_scene(
    scene: Inspire3DScene,
    out_path: str | Path,
    width: int = 900,
    height: int = 700,
    title: str | None = None,
    view_mode: str = "palm",
) -> None:
    image = _render_scene_image(
        scene,
        width=width,
        height=height,
        title=title or f"{scene.side} Inspire hand",
        view_mode=view_mode,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def render_inspire_3d_scenes(
    scenes: list[Inspire3DScene],
    out_path: str | Path,
    panel_width: int = 780,
    panel_height: int = 620,
    view_mode: str = "palm",
    title_suffix: str = "Inspire hand",
) -> None:
    from PIL import Image

    panels = [
        _render_scene_image(
            scene,
            width=panel_width,
            height=panel_height,
            title=f"{scene.side} {title_suffix}",
            view_mode=view_mode,
        )
        for scene in scenes
    ]
    image = Image.new("RGB", (panel_width * len(panels), panel_height), "white")
    for idx, panel in enumerate(panels):
        image.paste(panel, (idx * panel_width, 0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def export_sensor_points_3d(scenes: list[Inspire3DScene], out_path: str | Path) -> None:
    payload = {
        scene.side: {
            sensor.name: {
                "side": sensor.side,
                "semantic_region": sensor.semantic_region,
                "grid_shape": sensor.grid_shape,
                "taxel_indices": sensor.taxel_indices.astype(np.int64),
                "points": sensor.points.astype(np.float32),
            }
            for sensor in scene.sensors
        }
        for scene in scenes
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)


def _render_scene_image(scene: Inspire3DScene, width: int, height: int, title: str, view_mode: str) -> "Image.Image":
    from PIL import Image, ImageDraw

    margin = 42
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")

    all_points = [scene.vertices] if scene.vertices.size else []
    all_points.extend(sensor.points for sensor in scene.sensors)
    bounds_points = np.concatenate(all_points, axis=0)
    view_sign = _view_sign(scene.side, view_mode)
    projected, _ = _project_points(bounds_points, view_sign)
    xy_min = projected.min(axis=0)
    xy_max = projected.max(axis=0)
    extent = np.maximum(xy_max - xy_min, 1e-9)
    scale = min((width - 2 * margin) / extent[0], (height - 2 * margin) / extent[1])
    center = (xy_min + xy_max) / 2.0

    def to_screen(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy, z = _project_points(points, view_sign)
        screen = np.empty_like(xy)
        screen[:, 0] = (xy[:, 0] - center[0]) * scale + width / 2.0
        screen[:, 1] = height / 2.0 - (xy[:, 1] - center[1]) * scale
        return screen, z

    if scene.vertices.size and scene.faces.size:
        vertex_xy, vertex_depth = to_screen(scene.vertices)
        face_depth = vertex_depth[scene.faces].mean(axis=1)
        for face_id in np.argsort(face_depth):
            polygon = [(float(x), float(y)) for x, y in vertex_xy[scene.faces[int(face_id)]]]
            shade = _face_shade(scene.vertices[scene.faces[int(face_id)]], view_sign)
            draw.polygon(polygon, fill=(shade, shade, shade, 72), outline=(105, 105, 105, 36))

    sensor_color = (35, 110, 235) if scene.side == "left" else (235, 82, 45)
    for sensor in scene.sensors:
        xy, _ = to_screen(sensor.points)
        rows, cols = sensor.grid_shape
        grid = xy.reshape(rows, cols, 2)
        line_color = sensor_color + (135,)
        for row in range(rows):
            draw.line([(float(x), float(y)) for x, y in grid[row]], fill=line_color, width=1)
        for col in range(cols):
            draw.line([(float(x), float(y)) for x, y in grid[:, col]], fill=line_color, width=1)
        radius = 2 if rows * cols > 9 else 3
        for x, y in xy:
            draw.ellipse(
                (float(x - radius), float(y - radius), float(x + radius), float(y + radius)),
                fill=sensor_color + (235,),
                outline=(255, 255, 255, 180),
            )

    draw.text((12, 10), title, fill=(20, 20, 20, 255))
    draw.text(
        (12, height - 26),
        f"collision mesh in base frame + sensor taxel centers | view={view_mode}",
        fill=(95, 95, 95, 255),
    )
    return image.convert("RGB")


def _view_sign(side: str, view_mode: str) -> float:
    if view_mode == "palm":
        return -1.0 if side == "left" else 1.0
    if view_mode == "same-camera":
        return 1.0
    raise ValueError("view_mode must be 'palm' or 'same-camera'")


def _project_points(points: np.ndarray, view_sign: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0] + 0.85 * view_sign * points[:, 1]
    y = points[:, 2] - 0.12 * view_sign * points[:, 1]
    depth = view_sign * points[:, 1]
    return np.stack([x, y], axis=1), depth


def _face_shade(triangle: np.ndarray, view_sign: float) -> int:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return 185
    normal = normal / norm
    light = np.array([0.35, 0.25 * view_sign, 0.90], dtype=np.float64)
    light = light / np.linalg.norm(light)
    value = 0.55 + 0.45 * abs(float(np.dot(normal, light)))
    return int(165 + 55 * value)


def _origin_transform(origin_el: ET.Element | None) -> np.ndarray:
    if origin_el is None:
        return np.eye(4, dtype=np.float64)
    xyz = _parse_vector(origin_el.attrib.get("xyz", "0 0 0"))
    rpy = _parse_vector(origin_el.attrib.get("rpy", "0 0 0"))
    return transform_matrix(xyz, rpy)


def _geometry_mesh(geometry_el: ET.Element, root_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    mesh_el = geometry_el.find("mesh")
    if mesh_el is not None:
        filename = mesh_el.attrib.get("filename")
        if not filename:
            return None
        mesh_path = _resolve_mesh_path(filename, root_dir)
        suffix = mesh_path.suffix.lower()
        if suffix == ".obj":
            vertices, faces = _read_obj_mesh(mesh_path)
        elif suffix == ".stl":
            vertices, faces = _read_stl_mesh(mesh_path)
        else:
            return None
        scale = _parse_vector(mesh_el.attrib.get("scale", "1 1 1"))
        return vertices * np.asarray(scale, dtype=np.float64), faces

    box_el = geometry_el.find("box")
    if box_el is not None:
        return _box_mesh(_parse_vector(box_el.attrib["size"]))

    cylinder_el = geometry_el.find("cylinder")
    if cylinder_el is not None:
        return _cylinder_mesh(
            radius=float(cylinder_el.attrib["radius"]),
            length=float(cylinder_el.attrib["length"]),
        )

    return None


def _resolve_mesh_path(filename: str, root_dir: Path) -> Path:
    if filename.startswith("package://"):
        package_path = filename[len("package://") :]
        parts = package_path.split("/", 1)
        if len(parts) == 1:
            return root_dir.parent / parts[0]
        return root_dir.parent / parts[1]
    return root_dir / filename


def _read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                indexes = [_obj_vertex_index(token) for token in parts[1:]]
                for i in range(1, len(indexes) - 1):
                    faces.append([indexes[0], indexes[i], indexes[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _read_stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack("<I", data[80:84])[0]
        expected = 84 + tri_count * 50
        if expected == len(data):
            return _read_binary_stl_mesh(data, tri_count)
    return _read_ascii_stl_mesh(path)


def _read_binary_stl_mesh(data: bytes, tri_count: int) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    offset = 84
    for _ in range(tri_count):
        record = struct.unpack("<12fH", data[offset : offset + 50])
        offset += 50
        face = []
        for idx in range(3):
            start = 3 + idx * 3
            face.append(len(vertices))
            vertices.append([float(record[start]), float(record[start + 1]), float(record[start + 2])])
        faces.append(face)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _read_ascii_stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    current: list[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                current.append(len(vertices))
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(current) == 3:
                    faces.append(current)
                    current = []
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _obj_vertex_index(token: str) -> int:
    return int(token.split("/")[0]) - 1


def _box_mesh(size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = [float(v) / 2.0 for v in size]
    vertices = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _cylinder_mesh(radius: float, length: float, segments: int = 28) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    z0 = -float(length) / 2.0
    z1 = float(length) / 2.0
    bottom = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, z0)], axis=1)
    top = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, z1)], axis=1)
    vertices = np.concatenate([bottom, top, [[0.0, 0.0, z0], [0.0, 0.0, z1]]], axis=0).astype(np.float64)
    bottom_center = 2 * segments
    top_center = 2 * segments + 1
    faces: list[list[int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
        faces.append([bottom_center, j, i])
        faces.append([top_center, segments + i, segments + j])
    return vertices, np.asarray(faces, dtype=np.int64)


def _parse_vector(raw: str) -> tuple[float, float, float]:
    values = [float(v) for v in raw.split()]
    if len(values) == 1:
        return values[0], values[0], values[0]
    if len(values) != 3:
        raise ValueError(f"expected 3-vector, got {raw!r}")
    return values[0], values[1], values[2]
