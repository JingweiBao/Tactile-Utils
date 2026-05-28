from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Iterable

import numpy as np

from sub_modules.offline_shape_alignment.alignment import apply_similarity_to_points
from sub_modules.offline_shape_alignment.types import Mesh


def render_alignment_report(
    report: dict,
    robot_mesh: Mesh,
    mano_mesh: Mesh,
    out_path: str | Path,
    *,
    width: int = 980,
    height: int = 740,
) -> None:
    image = _render_panel(report, robot_mesh, mano_mesh, width=width, height=height)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def render_alignment_reports(
    panels: Iterable[tuple[dict, Mesh, Mesh]],
    out_path: str | Path,
    *,
    panel_width: int = 760,
    panel_height: int = 620,
) -> None:
    from PIL import Image

    panel_images = [
        _render_panel(report, robot_mesh, mano_mesh, width=panel_width, height=panel_height)
        for report, robot_mesh, mano_mesh in panels
    ]
    if not panel_images:
        raise ValueError("at least one panel is required")
    image = Image.new("RGB", (panel_width * len(panel_images), panel_height), "white")
    for idx, panel in enumerate(panel_images):
        image.paste(panel, (idx * panel_width, 0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def render_loss_histories(
    histories: dict[str, list[dict[str, float]]],
    out_path: str | Path,
    *,
    width: int = 980,
    height: int = 520,
) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin_left = 64
    margin_right = 28
    margin_top = 34
    margin_bottom = 54
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_steps = []
    all_values = []
    for history in histories.values():
        for item in history:
            if "total" in item:
                all_steps.append(float(item["step"]))
                all_values.append(max(float(item["total"]), 1e-12))
    if not all_steps or not all_values:
        raise ValueError("loss history is empty")

    x_min, x_max = 0.0, max(all_steps)
    y_min = float(np.floor(np.log10(min(all_values))))
    y_max = float(np.ceil(np.log10(max(all_values))))
    if y_min == y_max:
        y_max = y_min + 1.0

    def xy(step: float, value: float) -> tuple[float, float]:
        x = margin_left + (float(step) - x_min) / max(1e-9, x_max - x_min) * plot_w
        y_value = np.log10(max(float(value), 1e-12))
        y = margin_top + (y_max - y_value) / max(1e-9, y_max - y_min) * plot_h
        return x, y

    draw.rectangle((margin_left, margin_top, margin_left + plot_w, margin_top + plot_h), outline=(210, 210, 210))
    for tick in range(int(y_min), int(y_max) + 1):
        y = margin_top + (y_max - tick) / max(1e-9, y_max - y_min) * plot_h
        draw.line((margin_left, y, margin_left + plot_w, y), fill=(235, 235, 235))
        draw.text((12, y - 7), f"1e{tick}", fill=(90, 90, 90))

    palette = [(32, 93, 180), (216, 107, 42), (70, 145, 96), (125, 83, 178)]
    for idx, (label, history) in enumerate(histories.items()):
        color = palette[idx % len(palette)]
        points = [xy(item["step"], item["total"]) for item in history if "total" in item]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        final = history[-1]
        legend_y = margin_top + idx * 18
        draw.line((width - 220, legend_y + 6, width - 190, legend_y + 6), fill=color, width=3)
        draw.text((width - 184, legend_y), f"{label}: {final['total']:.3g}", fill=(45, 45, 45))

    draw.text((12, 10), "MANO beta-only shape optimization loss", fill=(20, 20, 20))
    draw.text((margin_left, height - 32), "step", fill=(90, 90, 90))
    draw.text((12, height - 32), "total loss (log10)", fill=(90, 90, 90))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _render_panel(report: dict, robot_mesh: Mesh, mano_mesh: Mesh, *, width: int, height: int):
    from PIL import Image, ImageDraw

    margin = 46
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")

    transform = np.asarray(report["transform"]["robot_to_mano"], dtype=np.float64)
    robot_vertices = apply_similarity_to_points(transform, robot_mesh.vertices)
    mano_vertices = np.asarray(mano_mesh.vertices, dtype=np.float64)
    robot_keypoints = np.asarray([item["robot_aligned"] for item in report["keypoints"]], dtype=np.float64)
    mano_keypoints = np.asarray([item["mano"] for item in report["keypoints"]], dtype=np.float64)

    bounds_clouds = [mano_vertices, robot_vertices[::_mesh_stride(robot_vertices.shape[0], 60000)], robot_keypoints, mano_keypoints]
    bounds_points = np.concatenate([points for points in bounds_clouds if points.size], axis=0)
    projected, _ = _project_points(bounds_points)
    xy_min = projected.min(axis=0)
    xy_max = projected.max(axis=0)
    extent = np.maximum(xy_max - xy_min, 1e-9)
    scale = min((width - 2 * margin) / extent[0], (height - 2 * margin - 44) / extent[1])
    center = (xy_min + xy_max) / 2.0

    def to_screen(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy, depth = _project_points(points)
        screen = np.empty_like(xy)
        screen[:, 0] = (xy[:, 0] - center[0]) * scale + width / 2.0
        screen[:, 1] = height / 2.0 - 8.0 - (xy[:, 1] - center[1]) * scale
        return screen, depth

    _draw_mesh(draw, mano_vertices, mano_mesh.faces, to_screen, fill=(95, 130, 190, 46), outline=(55, 78, 118, 42), max_faces=1538)
    _draw_mesh(draw, robot_vertices, robot_mesh.faces, to_screen, fill=(235, 130, 65, 38), outline=(160, 82, 34, 32), max_faces=22000)

    robot_xy, _ = to_screen(robot_keypoints)
    mano_xy, _ = to_screen(mano_keypoints)
    for idx, label in enumerate(report["labels"]):
        line_color = (45, 130, 88, 95)
        draw.line(
            [(float(robot_xy[idx, 0]), float(robot_xy[idx, 1])), (float(mano_xy[idx, 0]), float(mano_xy[idx, 1]))],
            fill=line_color,
            width=1,
        )
        _draw_point(draw, robot_xy[idx], color=(225, 96, 40, 230), radius=3)
        _draw_point(draw, mano_xy[idx], color=(42, 91, 180, 235), radius=3)
        if label.endswith("_tip") or label == "wrist":
            draw.text((float(mano_xy[idx, 0] + 4), float(mano_xy[idx, 1] - 4)), label, fill=(30, 30, 30, 220))

    summary = report["summary"]
    status = report["status"]
    title = f"{report['side']} XHand -> MANO reference diagnostic"
    footer = (
        f"mean={summary['mean_distance_mm']:.1f}mm  max={summary['max_distance_mm']:.1f}mm  "
        f"scale={summary['scale']:.3f}  pose={status['pose']['level']}  "
        f"mirror={status['mirror']['level']}  unit={status['unit']['level']}"
    )
    draw.text((12, 10), title, fill=(20, 20, 20, 255))
    draw.text((12, height - 26), footer, fill=(82, 82, 82, 255))
    return image.convert("RGB")


def _draw_mesh(draw, vertices: np.ndarray, faces: np.ndarray, to_screen, *, fill, outline, max_faces: int) -> None:
    if vertices.size == 0 or faces.size == 0:
        return
    face_step = max(1, int(ceil(faces.shape[0] / max_faces)))
    sampled_faces = faces[::face_step]
    xy, depth = to_screen(vertices)
    face_depth = depth[sampled_faces].mean(axis=1)
    for face_id in np.argsort(face_depth):
        face = sampled_faces[int(face_id)]
        polygon = [(float(x), float(y)) for x, y in xy[face]]
        draw.polygon(polygon, fill=fill, outline=outline)


def _draw_point(draw, xy: np.ndarray, *, color: tuple[int, int, int, int], radius: int) -> None:
    x, y = float(xy[0]), float(xy[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255, 210))


def _project_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0] + 0.72 * points[:, 1]
    y = points[:, 2] - 0.16 * points[:, 1]
    depth = points[:, 1]
    return np.stack([x, y], axis=1), depth


def _mesh_stride(count: int, max_points: int) -> int:
    return max(1, int(ceil(count / max_points)))
