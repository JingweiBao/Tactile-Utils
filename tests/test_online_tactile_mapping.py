from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from online.cli import main as online_main
from online.tactile_mapping import (
    MANO_FULL_POSE_LABELS,
    MANO_POSE_LABELS,
    axis_angle_to_matrices,
    keypoints_to_mano_pose,
    load_online_reference,
    map_taxels_to_posed_surface,
    merge_qpos_with_reference,
    project_online_tactile,
)
from online.retarget_calibration import (
    fit_ridge_retarget_model,
    load_retarget_model,
    predict_retarget_pose,
    save_retarget_model,
)
from offline.tactile_to_mano_projection import apply_sensor_values_to_vertices
from sub_modules.offline_shape_alignment.xhand import default_xhand_urdf_path


ROOT = Path(__file__).resolve().parents[1]
MANO_ROOT = ROOT / "assets" / "hands" / "mano"
XHAND_ROOT = ROOT / "assets" / "hands" / "xhand"
XHAND_RIGHT_URDF = default_xhand_urdf_path(XHAND_ROOT, "right")
_TORCH_IMPORT_ERROR: str | None | bool = None


def _skip_without_assets(testcase: unittest.TestCase) -> None:
    required = [
        MANO_ROOT / "mano_v1_2" / "models" / "MANO_RIGHT.pkl",
        XHAND_RIGHT_URDF,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        testcase.skipTest(f"online tactile mapping assets are not available: {missing}")


def _skip_without_torch(testcase: unittest.TestCase) -> None:
    global _TORCH_IMPORT_ERROR
    if _TORCH_IMPORT_ERROR is None:
        try:
            subprocess.run(
                [sys.executable, "-c", "import torch"],
                check=True,
                timeout=15.0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            _TORCH_IMPORT_ERROR = False
        except subprocess.TimeoutExpired:
            _TORCH_IMPORT_ERROR = "PyTorch import timed out"
        except Exception as exc:
            _TORCH_IMPORT_ERROR = str(exc)
    if _TORCH_IMPORT_ERROR:
        testcase.skipTest(f"PyTorch is not available for online MANO pose tests: {_TORCH_IMPORT_ERROR}")


class OnlineTactileMappingTest(unittest.TestCase):
    def test_barycentric_mapping_recomputes_taxel_points_on_posed_vertices(self) -> None:
        vertices = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        nearest_faces = np.asarray([0, 1], dtype=np.int64)
        barycentric = np.asarray([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], dtype=np.float64)

        points = map_taxels_to_posed_surface(vertices, faces, nearest_faces, barycentric)

        self.assertTrue(np.allclose(points[0], [0.25, 0.25, 1.0]))
        self.assertTrue(np.allclose(points[1], [0.5, 0.8, 1.0]))

    def test_sparse_weights_map_tactile_values_to_vertices(self) -> None:
        rows = np.asarray([0, 1, 2, 1, 2, 3], dtype=np.int64)
        cols = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        vals = np.asarray([0.5, 0.25, 0.25, 0.2, 0.3, 0.5], dtype=np.float64)
        tactile = np.asarray([2.0, 4.0], dtype=np.float64)

        vertex_values = apply_sensor_values_to_vertices(rows, cols, vals, (4, 2), tactile)

        self.assertTrue(np.allclose(vertex_values, [2.0, 2.888888888888889, 3.090909090909091, 4.0]))

    def test_missing_qpos_falls_back_to_reference_qpos(self) -> None:
        merged = merge_qpos_with_reference(
            {"joint_a": 0.1, "joint_b": 0.2},
            {"joint_b": 0.7, "joint_c": -0.3},
        )

        self.assertEqual(merged, {"joint_a": 0.1, "joint_b": 0.7, "joint_c": -0.3})

    def test_identity_keypoint_retargeting_returns_base_pose(self) -> None:
        keypoints = _toy_keypoints()
        base_pose = np.zeros((len(MANO_POSE_LABELS), 3), dtype=np.float64)
        base_pose[0] = [0.05, -0.02, 0.01]
        base_pose[-1] = [-0.01, 0.03, 0.02]

        pose = keypoints_to_mano_pose(keypoints, keypoints, base_pose)

        self.assertTrue(np.allclose(pose, base_pose, atol=1e-7))

    def test_keypoint_retargeting_can_return_root_pose(self) -> None:
        keypoints = _toy_keypoints()
        angle = 0.37
        root_rotation = axis_angle_to_matrices(np.asarray([[0.0, 0.0, angle]], dtype=np.float64))[0]
        rotated = {label: root_rotation @ point for label, point in keypoints.items()}
        base_pose = np.zeros((len(MANO_POSE_LABELS), 3), dtype=np.float64)
        base_pose[2] = [0.01, -0.02, 0.03]

        pose = keypoints_to_mano_pose(keypoints, rotated, base_pose, include_root=True)

        self.assertEqual(pose.shape, (len(MANO_FULL_POSE_LABELS), 3))
        self.assertTrue(np.allclose(axis_angle_to_matrices(pose[:1])[0], root_rotation, atol=1e-7))
        self.assertTrue(np.allclose(pose[1:], base_pose, atol=1e-7))

    def test_ridge_retarget_model_predicts_qpos_to_pose(self) -> None:
        qpos_names = ("joint_a", "joint_b")
        qpos = np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.25],
            ],
            dtype=np.float64,
        )
        poses = np.zeros((qpos.shape[0], len(MANO_FULL_POSE_LABELS), 3), dtype=np.float64)
        poses[:, 1, 0] = 0.2 + 0.4 * qpos[:, 0]
        poses[:, 4, 1] = -0.1 + 0.3 * qpos[:, 1]
        model = fit_ridge_retarget_model("right", qpos_names, qpos, poses, ridge_lambda=1e-10)

        prediction = predict_retarget_pose(model, qpos)

        self.assertTrue(np.allclose(prediction, poses, atol=1e-7))
        self.assertTrue(np.allclose(model.predict({"joint_a": 0.5, "joint_b": 0.0}), poses[1], atol=1e-7))

    def test_retarget_model_save_and_load_round_trips(self) -> None:
        qpos_names = ("joint_a",)
        qpos = np.asarray([[0.0], [1.0]], dtype=np.float64)
        poses = np.zeros((2, len(MANO_FULL_POSE_LABELS), 3), dtype=np.float64)
        poses[:, 2, 2] = qpos[:, 0]
        model = fit_ridge_retarget_model("right", qpos_names, qpos, poses)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "retarget.npz"
            save_retarget_model(model, path)
            loaded = load_retarget_model(path)

        self.assertEqual(loaded.qpos_names, qpos_names)
        self.assertTrue(np.allclose(predict_retarget_pose(loaded, qpos), predict_retarget_pose(model, qpos)))

    def test_online_reference_and_projection_smoke_with_assets(self) -> None:
        _skip_without_assets(self)
        _skip_without_torch(self)

        with tempfile.TemporaryDirectory() as tmpdir:
            alignment_path = _write_alignment_json(Path(tmpdir), "right")
            projection_path = _write_projection_npz(Path(tmpdir), vertex_count=778, face_count=1538, taxel_count=2)
            reference = load_online_reference(
                alignment_path,
                projection_path,
                "right",
                xhand_root=XHAND_ROOT,
                mano_root=MANO_ROOT,
            )
            result = project_online_tactile(reference, {}, np.asarray([0.25, 0.75], dtype=np.float64))

        self.assertEqual(result.mano_mesh.vertices.shape, (778, 3))
        self.assertEqual(result.vertex_tactile_values.shape, (778,))
        self.assertEqual(result.taxel_surface_points.shape, (2, 3))
        self.assertEqual(result.mano_pose.shape, (16, 3))

    def test_online_cli_smoke_writes_npz_json_and_png(self) -> None:
        _skip_without_assets(self)
        _skip_without_torch(self)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            alignment_path = _write_alignment_json(tmp, "right")
            projection_path = _write_projection_npz(tmp, vertex_count=778, face_count=1538, taxel_count=2)
            qpos_path = tmp / "qpos.json"
            tactile_path = tmp / "tactile.npy"
            qpos_path.write_text(json.dumps({"qpos": {}}), encoding="utf-8")
            np.save(tactile_path, np.asarray([1.0, 0.5], dtype=np.float64))

            online_main(
                [
                    "map-xhand-tactile",
                    "--side",
                    "right",
                    "--alignment-json",
                    str(alignment_path),
                    "--projection-npz",
                    str(projection_path),
                    "--qpos-json",
                    str(qpos_path),
                    "--tactile-values",
                    str(tactile_path),
                    "--xhand-root",
                    str(XHAND_ROOT),
                    "--mano-root",
                    str(MANO_ROOT),
                    "--results-root",
                    str(tmp),
                    "--timestamp",
                    "20260531_120000",
                    "--out-json",
                    "online.json",
                    "--out-npz",
                    "online.npz",
                    "--out-png",
                    "online.png",
                ]
            )
            base = tmp / "online_tactile_mapping"
            json_path = base / "20260531_120000_online.json"
            npz_path = base / "20260531_120000_online.npz"
            png_path = base / "20260531_120000_online.png"

            self.assertTrue(json_path.exists())
            self.assertTrue(npz_path.exists())
            self.assertTrue(png_path.exists())
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            with np.load(npz_path) as result_npz:
                vertices_shape = result_npz["mano_vertices"].shape
                transform_shape = result_npz["mano_frame_transform"].shape

        self.assertEqual(payload["side"], "right")
        self.assertEqual(payload["taxel_count"], 2)
        self.assertEqual(vertices_shape, (778, 3))
        self.assertEqual(transform_shape, (4, 4))


def _toy_keypoints() -> dict[str, np.ndarray]:
    keypoints = {"wrist": np.asarray([0.0, -1.0, 0.0], dtype=np.float64)}
    xs = {
        "thumb": -0.8,
        "index": -0.3,
        "middle": 0.0,
        "ring": 0.3,
        "pinky": 0.6,
    }
    for finger, x in xs.items():
        keypoints[f"{finger}_mcp"] = np.asarray([x, 0.0, 0.0], dtype=np.float64)
        keypoints[f"{finger}_pip"] = np.asarray([x, 1.0, 0.1], dtype=np.float64)
        keypoints[f"{finger}_dip"] = np.asarray([x, 2.0, 0.1], dtype=np.float64)
        keypoints[f"{finger}_tip"] = np.asarray([x, 3.0, 0.0], dtype=np.float64)
    return keypoints


def _write_alignment_json(tmp: Path, side: str) -> Path:
    path = tmp / "alignment.json"
    payload = {
        "module": "offline_shape_alignment",
        "command": "fit-mano-shape-pose",
        "sides": {
            side: {
                "side": side,
                "beta": [0.0] * 10,
                "pose_residual_labels": list(MANO_POSE_LABELS),
                "pose_residual": np.zeros((len(MANO_POSE_LABELS), 3), dtype=np.float64).tolist(),
                "robot_qpos": {},
                "fixed_frame_report": {
                    "transform": {
                        "robot_to_mano": np.eye(4, dtype=np.float64).tolist(),
                    },
                    "keypoints": [
                        {
                            "label": label,
                            "mano": point.tolist(),
                        }
                        for label, point in _toy_keypoints().items()
                    ],
                },
            }
        },
        "outputs": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_projection_npz(tmp: Path, *, vertex_count: int, face_count: int, taxel_count: int) -> Path:
    path = tmp / "projection.npz"
    rows = np.asarray([0, 1, 2, 1, 2, 3][: taxel_count * 3], dtype=np.int64)
    cols = np.repeat(np.arange(taxel_count, dtype=np.int64), 3)
    vals = np.tile(np.asarray([0.5, 0.25, 0.25], dtype=np.float64), taxel_count)
    nearest = np.arange(taxel_count, dtype=np.int64) % face_count
    bary = np.tile(np.asarray([[0.5, 0.25, 0.25]], dtype=np.float64), (taxel_count, 1))
    valid = np.zeros(vertex_count, dtype=np.bool_)
    valid[np.unique(rows)] = True
    np.savez_compressed(
        path,
        rows=rows,
        cols=cols,
        vals=vals,
        shape=np.asarray([vertex_count, taxel_count], dtype=np.int64),
        nearest_face_indices=nearest,
        barycentric=bary,
        valid_vertex_mask=valid,
        sensor_names=np.asarray(["right_index_tactile_sensor"] * taxel_count),
        semantic_regions=np.asarray(["right_index"] * taxel_count),
        projection_mode=np.asarray("test"),
    )
    return path


if __name__ == "__main__":
    unittest.main()
