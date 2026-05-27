from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offline_shape_alignment.types import Mesh


@dataclass(frozen=True)
class SurfaceSamplePattern:
    face_indices: np.ndarray
    barycentric: np.ndarray


def make_surface_sample_pattern(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    *,
    seed: int = 0,
) -> SurfaceSamplePattern:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    count = int(count)
    if count <= 0:
        raise ValueError(f"surface sample count must be positive, got {count}")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"expected vertices with shape Nx3, got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"expected triangular faces with shape Fx3, got {faces.shape}")

    areas = triangle_areas(vertices, faces)
    total_area = float(areas.sum())
    if total_area <= 1e-18:
        raise ValueError("mesh has no positive-area faces")

    rng = np.random.default_rng(seed)
    face_indices = rng.choice(faces.shape[0], size=count, replace=True, p=areas / total_area)
    barycentric = random_barycentric(count, rng)
    return SurfaceSamplePattern(
        face_indices=face_indices.astype(np.int64),
        barycentric=barycentric.astype(np.float64),
    )


def sample_mesh_surface(mesh: Mesh, count: int, *, seed: int = 0) -> np.ndarray:
    pattern = make_surface_sample_pattern(mesh.vertices, mesh.faces, count, seed=seed)
    return apply_surface_sample_pattern(mesh.vertices, mesh.faces, pattern)


def apply_surface_sample_pattern(
    vertices: np.ndarray,
    faces: np.ndarray,
    pattern: SurfaceSamplePattern,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri_vertices = vertices[faces[np.asarray(pattern.face_indices, dtype=np.int64)]]
    barycentric = np.asarray(pattern.barycentric, dtype=np.float64)
    return np.sum(tri_vertices * barycentric[:, :, None], axis=1)


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def random_barycentric(count: int, rng: np.random.Generator) -> np.ndarray:
    uv = rng.random((int(count), 2), dtype=np.float64)
    sqrt_u = np.sqrt(uv[:, 0])
    return np.stack(
        [
            1.0 - sqrt_u,
            sqrt_u * (1.0 - uv[:, 1]),
            sqrt_u * uv[:, 1],
        ],
        axis=1,
    )
