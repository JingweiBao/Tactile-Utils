from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import pickle
import sys
import types

import numpy as np

from offline_shape_alignment.types import KEYPOINT_LABELS, KeypointSet, Mesh


MANO_JOINT_INDEXES: dict[str, tuple[int, int, int]] = {
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "pinky": (7, 8, 9),
    "ring": (10, 11, 12),
    "thumb": (13, 14, 15),
}


@dataclass(frozen=True)
class MANOReference:
    side: str
    model_path: Path
    mesh: Mesh
    joints: np.ndarray
    kintree_table: np.ndarray
    keypoints: KeypointSet
    fingertip_vertex_indices: dict[str, int]


def load_mano_reference(side: str, mano_root: str | Path) -> MANOReference:
    side = _validate_side(side)
    model_path = default_mano_model_path(mano_root, side)
    data = _load_mano_pickle(model_path)

    vertices = np.asarray(data["v_template"], dtype=np.float64)
    faces = np.asarray(data["f"], dtype=np.int64)
    joints = np.asarray(data["J"], dtype=np.float64)
    kintree_table = np.asarray(data["kintree_table"], dtype=np.int64)
    fingertip_vertex_indices = infer_mano_fingertip_vertices(vertices, joints)
    keypoints = _build_mano_keypoints(side, joints, vertices, fingertip_vertex_indices, model_path)

    return MANOReference(
        side=side,
        model_path=model_path,
        mesh=Mesh(vertices=vertices, faces=faces),
        joints=joints,
        kintree_table=kintree_table,
        keypoints=keypoints,
        fingertip_vertex_indices=fingertip_vertex_indices,
    )


def default_mano_model_path(mano_root: str | Path, side: str) -> Path:
    side = _validate_side(side)
    return Path(mano_root) / "mano_v1_2" / "models" / f"MANO_{side.upper()}.pkl"


def infer_mano_fingertip_vertices(vertices: np.ndarray, joints: np.ndarray) -> dict[str, int]:
    vertices = np.asarray(vertices, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    used: set[int] = set()
    out: dict[str, int] = {}

    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        mcp_idx, pip_idx, dip_idx = MANO_JOINT_INDEXES[finger]
        idx = _infer_fingertip_vertex(
            vertices,
            mcp=joints[mcp_idx],
            pip=joints[pip_idx],
            dip=joints[dip_idx],
            used=used,
        )
        used.add(idx)
        out[finger] = idx
    return out


def _build_mano_keypoints(
    side: str,
    joints: np.ndarray,
    vertices: np.ndarray,
    fingertip_vertex_indices: dict[str, int],
    model_path: Path,
) -> KeypointSet:
    points = {
        "wrist": joints[0],
        "thumb_mcp": joints[13],
        "thumb_pip": joints[14],
        "thumb_dip": joints[15],
        "thumb_tip": vertices[fingertip_vertex_indices["thumb"]],
        "index_mcp": joints[1],
        "index_pip": joints[2],
        "index_dip": joints[3],
        "index_tip": vertices[fingertip_vertex_indices["index"]],
        "middle_mcp": joints[4],
        "middle_pip": joints[5],
        "middle_dip": joints[6],
        "middle_tip": vertices[fingertip_vertex_indices["middle"]],
        "ring_mcp": joints[10],
        "ring_pip": joints[11],
        "ring_dip": joints[12],
        "ring_tip": vertices[fingertip_vertex_indices["ring"]],
        "pinky_mcp": joints[7],
        "pinky_pip": joints[8],
        "pinky_dip": joints[9],
        "pinky_tip": vertices[fingertip_vertex_indices["pinky"]],
    }
    ordered = np.stack([points[label] for label in KEYPOINT_LABELS], axis=0).astype(np.float64)
    return KeypointSet(
        side=side,
        labels=KEYPOINT_LABELS,
        points=ordered,
        metadata={
            "hand": "mano",
            "model_path": str(model_path),
            "pose": "template_zero_pose",
            "fingertip_vertex_indices": fingertip_vertex_indices,
        },
    )


def _infer_fingertip_vertex(
    vertices: np.ndarray,
    *,
    mcp: np.ndarray,
    pip: np.ndarray,
    dip: np.ndarray,
    used: set[int],
) -> int:
    axis = np.asarray(dip, dtype=np.float64) - np.asarray(pip, dtype=np.float64)
    if float(np.linalg.norm(axis)) < 1e-9:
        axis = np.asarray(dip, dtype=np.float64) - np.asarray(mcp, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("cannot infer MANO fingertip from degenerate finger joints")
    axis = axis / norm

    rel = vertices - np.asarray(dip, dtype=np.float64)
    projection = rel @ axis
    perpendicular = np.linalg.norm(rel - projection[:, None] * axis[None, :], axis=1)

    forward = projection > -0.004
    close_to_ray = perpendicular < 0.025
    candidates = np.nonzero(forward & close_to_ray)[0]
    if candidates.size == 0:
        candidates = np.nonzero(forward)[0]
    if candidates.size == 0:
        candidates = np.arange(vertices.shape[0])

    score = projection[candidates] - 0.20 * perpendicular[candidates]
    if used:
        for pos, vertex_idx in enumerate(candidates):
            if int(vertex_idx) in used:
                score[pos] -= 1e6
    return int(candidates[int(np.argmax(score))])


def _load_mano_pickle(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return pickle.load(f, encoding="latin1")
    except ModuleNotFoundError:
        with _mano_pickle_stubs():
            with path.open("rb") as f:
                return pickle.load(f, encoding="latin1")


@contextmanager
def _mano_pickle_stubs():
    module_names = [
        "chumpy",
        "chumpy.ch",
        "chumpy.reordering",
        "scipy",
        "scipy.sparse",
        "scipy.sparse.csc",
    ]
    previous = {name: sys.modules.get(name) for name in module_names}
    try:
        if importlib.util.find_spec("chumpy") is None:
            chumpy = types.ModuleType("chumpy")
            chumpy_ch = types.ModuleType("chumpy.ch")
            chumpy_reordering = types.ModuleType("chumpy.reordering")
            chumpy_ch.Ch = _FakeCh
            chumpy_reordering.Select = _FakeSelect
            sys.modules["chumpy"] = chumpy
            sys.modules["chumpy.ch"] = chumpy_ch
            sys.modules["chumpy.reordering"] = chumpy_reordering

        if importlib.util.find_spec("scipy") is None:
            scipy = types.ModuleType("scipy")
            scipy_sparse = types.ModuleType("scipy.sparse")
            scipy_sparse_csc = types.ModuleType("scipy.sparse.csc")
            scipy_sparse_csc.csc_matrix = _FakeCscMatrix
            sys.modules["scipy"] = scipy
            sys.modules["scipy.sparse"] = scipy_sparse
            sys.modules["scipy.sparse.csc"] = scipy_sparse_csc
        elif importlib.util.find_spec("scipy.sparse.csc") is None:
            scipy_sparse_csc = types.ModuleType("scipy.sparse.csc")
            scipy_sparse_csc.csc_matrix = _FakeCscMatrix
            sys.modules["scipy.sparse.csc"] = scipy_sparse_csc
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _FakeCh:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class _FakeSelect:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class _FakeCscMatrix:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state

    @property
    def shape(self) -> tuple[int, int] | None:
        return self.__dict__.get("_shape")

    def toarray(self) -> np.ndarray:
        shape = self.shape
        if shape is None:
            raise ValueError("sparse matrix shape is unavailable")
        dense = np.zeros(shape, dtype=np.float64)
        indices = np.asarray(self.__dict__["indices"], dtype=np.int64)
        indptr = np.asarray(self.__dict__["indptr"], dtype=np.int64)
        data = np.asarray(self.__dict__["data"], dtype=np.float64)
        for col in range(shape[1]):
            start, end = int(indptr[col]), int(indptr[col + 1])
            dense[indices[start:end], col] = data[start:end]
        return dense


def _validate_side(side: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    return side
