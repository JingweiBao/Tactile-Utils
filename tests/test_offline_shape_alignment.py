from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from offline_shape_alignment.alignment import diagnose_alignment
from offline_shape_alignment.cli import main as offline_shape_alignment_main
from offline_shape_alignment.mano import load_mano_reference
from offline_shape_alignment.mano_torch import load_mano_beta_model
from offline_shape_alignment.reference_pose import fit_xhand_reference_pose
from offline_shape_alignment.render import render_alignment_report
from offline_shape_alignment.sampling import make_surface_sample_pattern, sample_mesh_surface
from offline_shape_alignment.shape_optimization import (
    PoseShapeOptimizationConfig,
    ShapeOptimizationConfig,
    fit_mano_beta_pose_to_xhand,
    fit_mano_beta_to_xhand,
)
from offline_shape_alignment.types import KEYPOINT_LABELS
from offline_shape_alignment.xhand import default_xhand_urdf_path, load_xhand_reference


ROOT = Path(__file__).resolve().parents[1]
MANO_ROOT = ROOT / "assets" / "hands" / "mano"
XHAND_ROOT = ROOT / "assets" / "hands" / "xhand"
XHAND_RIGHT_URDF = default_xhand_urdf_path(XHAND_ROOT, "right")
XHAND_LEFT_URDF = default_xhand_urdf_path(XHAND_ROOT, "left")


def _skip_without_assets(testcase: unittest.TestCase) -> None:
    required = [
        MANO_ROOT / "mano_v1_2" / "models" / "MANO_RIGHT.pkl",
        MANO_ROOT / "mano_v1_2" / "models" / "MANO_LEFT.pkl",
        XHAND_RIGHT_URDF,
        XHAND_LEFT_URDF,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        testcase.skipTest(f"offline shape alignment assets are not available: {missing}")


def _skip_without_torch(testcase: unittest.TestCase) -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        testcase.skipTest(f"PyTorch is not available for shape optimization tests: {exc}")


class OfflineShapeAlignmentTest(unittest.TestCase):
    def test_mano_reference_reader_loads_without_runtime_mano_dependencies(self) -> None:
        _skip_without_assets(self)

        ref = load_mano_reference("right", MANO_ROOT)

        self.assertEqual(ref.mesh.vertices.shape, (778, 3))
        self.assertEqual(ref.mesh.faces.shape, (1538, 3))
        self.assertEqual(ref.joints.shape, (16, 3))
        self.assertEqual(ref.keypoints.points.shape, (21, 3))
        self.assertEqual(ref.keypoints.labels, KEYPOINT_LABELS)
        self.assertEqual(set(ref.fingertip_vertex_indices), {"thumb", "index", "middle", "ring", "pinky"})
        self.assertEqual(len(set(ref.fingertip_vertex_indices.values())), 5)

    def test_xhand_keypoints_are_complete_for_both_sides(self) -> None:
        _skip_without_assets(self)

        for side in ("left", "right"):
            ref = load_xhand_reference(default_xhand_urdf_path(XHAND_ROOT, side), side)
            self.assertEqual(ref.keypoints.labels, KEYPOINT_LABELS)
            self.assertEqual(ref.keypoints.points.shape, (21, 3))
            self.assertGreater(ref.mesh.vertices.shape[0], 0)
            self.assertGreater(ref.mesh.faces.shape[0], 0)
            self.assertTrue(np.isfinite(ref.keypoints.points).all())
            self.assertEqual(ref.keypoints.metadata["root_link"], f"{side}_hand_link")

    def test_alignment_report_and_png_render(self) -> None:
        _skip_without_assets(self)

        robot = load_xhand_reference(XHAND_RIGHT_URDF, "right")
        mano = load_mano_reference("right", MANO_ROOT)
        report = diagnose_alignment(robot.keypoints, mano.keypoints, robot.mesh, mano.mesh)

        self.assertEqual(report["side"], "right")
        self.assertEqual(len(report["keypoints"]), 21)
        self.assertTrue(np.isfinite(np.asarray(report["transform"]["robot_to_mano"], dtype=np.float64)).all())
        self.assertGreater(report["summary"]["scale"], 0.0)
        self.assertGreaterEqual(report["summary"]["mean_distance_mm"], 0.0)
        self.assertIn(report["status"]["unit"]["level"], {"ok", "warn"})
        self.assertIn(report["status"]["mirror"]["level"], {"ok", "warn"})
        self.assertIn(report["status"]["pose"]["level"], {"ok", "warn"})

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "alignment.png"
            render_alignment_report(report, robot.mesh, mano.mesh, image_path, width=360, height=280)
            self.assertTrue(image_path.exists())
            self.assertGreater(image_path.stat().st_size, 0)

    def test_surface_sampling_is_deterministic(self) -> None:
        mesh = load_mano_reference("right", MANO_ROOT).mesh if (MANO_ROOT / "mano_v1_2" / "models" / "MANO_RIGHT.pkl").exists() else None
        if mesh is None:
            self.skipTest("MANO asset is not available")

        points_a = sample_mesh_surface(mesh, 32, seed=123)
        points_b = sample_mesh_surface(mesh, 32, seed=123)
        pattern = make_surface_sample_pattern(mesh.vertices, mesh.faces, 32, seed=123)

        self.assertTrue(np.allclose(points_a, points_b))
        self.assertEqual(pattern.face_indices.shape, (32,))
        self.assertEqual(pattern.barycentric.shape, (32, 3))
        self.assertTrue(np.allclose(pattern.barycentric.sum(axis=1), 1.0))

    def test_mano_beta_model_generates_shape_instance(self) -> None:
        _skip_without_assets(self)
        _skip_without_torch(self)

        model = load_mano_beta_model("right", MANO_ROOT)
        zero = model.instance()
        beta = np.zeros(model.num_betas, dtype=np.float64)
        beta[0] = 0.5
        shaped = model.instance(beta)

        self.assertEqual(model.num_betas, 10)
        self.assertEqual(zero.mesh.vertices.shape, (778, 3))
        self.assertEqual(zero.keypoints.labels, KEYPOINT_LABELS)
        self.assertFalse(np.allclose(zero.mesh.vertices, shaped.mesh.vertices))

    def test_mano_zero_pose_lbs_matches_beta_only_forward(self) -> None:
        _skip_without_assets(self)
        _skip_without_torch(self)

        model = load_mano_beta_model("right", MANO_ROOT)
        beta = np.zeros(model.num_betas, dtype=np.float64)
        beta[0] = 0.25
        beta_only = model.instance(beta)
        zero_pose = model.instance(beta, np.zeros((15, 3), dtype=np.float64))

        self.assertLess(float(np.max(np.abs(beta_only.mesh.vertices - zero_pose.mesh.vertices))), 1e-6)
        self.assertLess(float(np.max(np.abs(beta_only.keypoints.points - zero_pose.keypoints.points))), 1e-6)

    def test_xhand_reference_pose_fit_improves_or_preserves_objective(self) -> None:
        _skip_without_assets(self)

        mano = load_mano_reference("right", MANO_ROOT)
        fit = fit_xhand_reference_pose(XHAND_RIGHT_URDF, "right", mano.keypoints, iterations=2)

        self.assertGreater(len(fit.qpos), 0)
        self.assertLessEqual(fit.objective_m, fit.baseline_objective_m + 1e-12)
        self.assertTrue(np.isfinite(np.asarray(list(fit.qpos.values()), dtype=np.float64)).all())

        robot = load_xhand_reference(XHAND_RIGHT_URDF, "right", qpos=fit.qpos)
        report = diagnose_alignment(robot.keypoints, mano.keypoints, robot.mesh, mano.mesh)
        self.assertGreater(report["summary"]["scale"], 0.0)
        self.assertGreaterEqual(report["summary"]["mean_distance_mm"], 0.0)

    def test_cli_smoke_writes_json_and_png(self) -> None:
        _skip_without_assets(self)

        with tempfile.TemporaryDirectory() as tmpdir:
            offline_shape_alignment_main(
                [
                    "diagnose-xhand-reference",
                    "--side",
                    "right",
                    "--xhand-root",
                    str(XHAND_ROOT),
                    "--mano-root",
                    str(MANO_ROOT),
                    "--results-root",
                    tmpdir,
                    "--timestamp",
                    "20260527_180000",
                    "--out-json",
                    "xhand_right_diag.json",
                    "--out-png",
                    "xhand_right_diag.png",
                ]
            )
            json_path = Path(tmpdir) / "offline_shape_alignment" / "20260527_180000_xhand_right_diag.json"
            png_path = Path(tmpdir) / "offline_shape_alignment" / "20260527_180000_xhand_right_diag.png"
            self.assertTrue(json_path.exists())
            self.assertTrue(png_path.exists())
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertIn("right", payload["sides"])
        self.assertEqual(payload["sides"]["right"]["side"], "right")

    def test_fit_pose_cli_smoke_writes_json_pose_and_png(self) -> None:
        _skip_without_assets(self)

        with tempfile.TemporaryDirectory() as tmpdir:
            offline_shape_alignment_main(
                [
                    "fit-xhand-reference-pose",
                    "--side",
                    "right",
                    "--xhand-root",
                    str(XHAND_ROOT),
                    "--mano-root",
                    str(MANO_ROOT),
                    "--results-root",
                    tmpdir,
                    "--timestamp",
                    "20260527_181000",
                    "--iterations",
                    "1",
                    "--out-json",
                    "xhand_right_fit_diag.json",
                    "--out-png",
                    "xhand_right_fit_diag.png",
                    "--out-pose-json",
                    "xhand_right_pose.json",
                ]
            )
            base = Path(tmpdir) / "offline_shape_alignment"
            json_path = base / "20260527_181000_xhand_right_fit_diag.json"
            png_path = base / "20260527_181000_xhand_right_fit_diag.png"
            pose_path = base / "20260527_181000_xhand_right_pose.json"
            self.assertTrue(json_path.exists())
            self.assertTrue(png_path.exists())
            self.assertTrue(pose_path.exists())
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            with pose_path.open("r", encoding="utf-8") as f:
                pose_payload = json.load(f)

        self.assertIn("reference_pose_fit", payload["sides"]["right"])
        self.assertIn("baseline_summary", payload["sides"]["right"])
        self.assertGreater(len(pose_payload["sides"]["right"]["fit"]["qpos"]), 0)

    def test_mano_beta_shape_optimizer_smoke(self) -> None:
        _skip_without_assets(self)
        _skip_without_torch(self)

        result = fit_mano_beta_to_xhand(
            "right",
            xhand_root=XHAND_ROOT,
            mano_root=MANO_ROOT,
            config=ShapeOptimizationConfig(iterations=2, robot_surface_points=64, mano_surface_points=64, log_every=1),
            reference_pose_iterations=1,
        )

        self.assertEqual(result.beta.shape, (10,))
        self.assertEqual(result.fitted_mano.mesh.vertices.shape, (778, 3))
        self.assertGreater(len(result.history), 0)
        self.assertTrue(np.isfinite(result.beta).all())
        self.assertGreaterEqual(result.fixed_frame_report["summary"]["mean_distance_mm"], 0.0)

    def test_mano_beta_pose_shape_optimizer_smoke(self) -> None:
        _skip_without_assets(self)
        _skip_without_torch(self)

        result = fit_mano_beta_pose_to_xhand(
            "right",
            xhand_root=XHAND_ROOT,
            mano_root=MANO_ROOT,
            config=PoseShapeOptimizationConfig(
                beta_init_iterations=2,
                iterations=2,
                robot_surface_points=64,
                mano_surface_points=64,
                log_every=1,
                pose_limit_rad=0.1,
            ),
            reference_pose_iterations=1,
        )

        self.assertEqual(result.beta.shape, (10,))
        self.assertEqual(result.pose.shape, (15, 3))
        self.assertLessEqual(float(np.max(np.abs(result.pose))), 0.1 + 1e-6)
        self.assertEqual(result.fitted_mano.mesh.vertices.shape, (778, 3))
        self.assertGreater(len(result.history), 0)
        self.assertGreaterEqual(result.fixed_frame_report["summary"]["mean_distance_mm"], 0.0)

    def test_mano_shape_cli_smoke_writes_json_png_loss_and_obj(self) -> None:
        _skip_without_assets(self)
        _skip_without_torch(self)

        with tempfile.TemporaryDirectory() as tmpdir:
            offline_shape_alignment_main(
                [
                    "fit-mano-shape",
                    "--side",
                    "right",
                    "--xhand-root",
                    str(XHAND_ROOT),
                    "--mano-root",
                    str(MANO_ROOT),
                    "--results-root",
                    tmpdir,
                    "--timestamp",
                    "20260527_182000",
                    "--iterations",
                    "2",
                    "--robot-surface-points",
                    "64",
                    "--mano-surface-points",
                    "64",
                    "--reference-pose-iterations",
                    "1",
                    "--out-json",
                    "xhand_right_mano_beta.json",
                    "--out-png",
                    "xhand_right_mano_beta.png",
                    "--out-loss-png",
                    "xhand_right_mano_beta_loss.png",
                    "--out-obj",
                    "xhand_right_fitted_mano.obj",
                ]
            )
            base = Path(tmpdir) / "offline_shape_alignment"
            json_path = base / "20260527_182000_xhand_right_mano_beta.json"
            png_path = base / "20260527_182000_xhand_right_mano_beta.png"
            loss_path = base / "20260527_182000_xhand_right_mano_beta_loss.png"
            obj_path = base / "20260527_182000_xhand_right_fitted_mano.obj"
            self.assertTrue(json_path.exists())
            self.assertTrue(png_path.exists())
            self.assertTrue(loss_path.exists())
            self.assertTrue(obj_path.exists())
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertIn("right", payload["sides"])
        self.assertEqual(len(payload["sides"]["right"]["beta"]), 10)

    def test_mano_shape_pose_cli_smoke_writes_json_png_loss_and_obj(self) -> None:
        _skip_without_assets(self)
        _skip_without_torch(self)

        with tempfile.TemporaryDirectory() as tmpdir:
            offline_shape_alignment_main(
                [
                    "fit-mano-shape-pose",
                    "--side",
                    "right",
                    "--xhand-root",
                    str(XHAND_ROOT),
                    "--mano-root",
                    str(MANO_ROOT),
                    "--results-root",
                    tmpdir,
                    "--timestamp",
                    "20260527_183000",
                    "--beta-init-iterations",
                    "2",
                    "--iterations",
                    "2",
                    "--robot-surface-points",
                    "64",
                    "--mano-surface-points",
                    "64",
                    "--reference-pose-iterations",
                    "1",
                    "--out-json",
                    "xhand_right_mano_beta_pose.json",
                    "--out-png",
                    "xhand_right_mano_beta_pose.png",
                    "--out-loss-png",
                    "xhand_right_mano_beta_pose_loss.png",
                    "--out-obj",
                    "xhand_right_fitted_mano_beta_pose.obj",
                ]
            )
            base = Path(tmpdir) / "offline_shape_alignment"
            json_path = base / "20260527_183000_xhand_right_mano_beta_pose.json"
            png_path = base / "20260527_183000_xhand_right_mano_beta_pose.png"
            loss_path = base / "20260527_183000_xhand_right_mano_beta_pose_loss.png"
            obj_path = base / "20260527_183000_xhand_right_fitted_mano_beta_pose.obj"
            self.assertTrue(json_path.exists())
            self.assertTrue(png_path.exists())
            self.assertTrue(loss_path.exists())
            self.assertTrue(obj_path.exists())
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertIn("right", payload["sides"])
        self.assertEqual(len(payload["sides"]["right"]["beta"]), 10)
        self.assertEqual(np.asarray(payload["sides"]["right"]["pose_residual"]).shape, (15, 3))
        self.assertEqual(len(payload["sides"]["right"]["pose_residual_labels"]), 15)


if __name__ == "__main__":
    unittest.main()
