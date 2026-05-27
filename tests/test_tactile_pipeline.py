from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import pickle

from tactile_layout_3d_projection.config import load_sensor_mount_config
from tactile_layout_3d_projection.inspire import (
    build_inspire_3d_scene,
    export_sensor_points_3d,
    render_inspire_3d_scene,
)
from tactile_layout_3d_projection.urdf import load_urdf
from tactile_layout_3d_projection.xhand import (
    _load_tactile_xml_points,
    _mujoco_tactile_points_to_urdf_local,
    build_xhand_3d_scene,
)
from tactile_utils_common.results import module_result_path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "inspire_hand_sensor_mount.yaml"
RIGHT_URDF = ROOT / "assets" / "hands" / "inspire_hand_dexsuite" / "inspire_hand_right.urdf"
XHAND_ROOT = ROOT / "assets" / "hands" / "xhand"
XHAND_RIGHT_URDF = XHAND_ROOT / "Xhand-urdf" / "xhand1_right(1)" / "urdf" / "xhand_right.urdf"
XHAND_TACTILE_DIR = XHAND_ROOT / "tactile sensor" / "xhand_tac"


class Tactile3DVisualizationTest(unittest.TestCase):
    def test_module_result_path_uses_module_folder_and_timestamp_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = module_result_path(
                "tactile_layout_3d_projection",
                "results/raw_name",
                results_root=tmpdir,
                timestamp="20260527_161500",
                default_suffix=".png",
            )

        self.assertEqual(
            path,
            Path(tmpdir) / "tactile_layout_3d_projection" / "20260527_161500_raw_name.png",
        )

    def test_inspire_mount_config_loads(self) -> None:
        config = load_sensor_mount_config(CONFIG)
        self.assertEqual(len(config.sensors), 34)
        by_name = {sensor.name: sensor for sensor in config.sensors}
        self.assertEqual(by_name["right_index_finger_pad_sensor"].parent_link, "index_proximal")
        self.assertEqual(by_name["right_index_finger_tip_sensor"].grid_shape, (3, 3))

    def test_urdf_parses_links_joints_and_mimics(self) -> None:
        model = load_urdf(RIGHT_URDF)
        self.assertIn("hand_base_link", model.links)
        self.assertIn("index_tip", model.links)
        self.assertEqual(model.validate_mesh_references(), [])
        self.assertEqual(len([joint for joint in model.joints if joint.joint_type == "revolute"]), 12)
        mimic_names = {joint.name for joint in model.joints if joint.mimic_joint}
        self.assertIn("index_intermediate_joint", mimic_names)
        transforms = model.link_transforms()
        self.assertIn("thumb_tip", transforms)
        self.assertEqual(transforms["thumb_tip"].shape, (4, 4))

    def test_inspire_3d_scene_exports_sensor_points(self) -> None:
        config = load_sensor_mount_config(CONFIG)
        scene = build_inspire_3d_scene(config, RIGHT_URDF, side="right")
        self.assertGreater(scene.vertices.shape[0], 0)
        self.assertGreater(scene.faces.shape[0], 0)
        self.assertEqual(len(scene.sensors), 17)
        self.assertEqual(sum(sensor.points.shape[0] for sensor in scene.sensors), 1062)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "right.png"
            points_path = Path(tmpdir) / "points.pkl"
            render_inspire_3d_scene(scene, image_path, width=320, height=240)
            export_sensor_points_3d([scene], points_path)
            image_exists = image_path.exists()
            with points_path.open("rb") as f:
                payload = pickle.load(f)

        self.assertTrue(image_exists)
        self.assertIn("right", payload)
        self.assertIn("right_palm_sensor", payload["right"])
        self.assertEqual(payload["right"]["right_palm_sensor"]["points"].shape, (112, 3))

    def test_xhand_3d_scene_reads_urdf_mesh_and_tactile_points(self) -> None:
        if not XHAND_RIGHT_URDF.exists() or not XHAND_TACTILE_DIR.exists():
            self.skipTest("XHand assets are not available")

        scene = build_xhand_3d_scene(XHAND_RIGHT_URDF, XHAND_TACTILE_DIR, side="right")
        self.assertGreater(scene.vertices.shape[0], 0)
        self.assertGreater(scene.faces.shape[0], 0)
        self.assertEqual(len(scene.sensors), 5)
        self.assertEqual(sum(sensor.points.shape[0] for sensor in scene.sensors), 600)
        self.assertEqual(scene.sensors[0].grid_shape, (120, 1))
        self.assertTrue(all(sensor.side == "right" for sensor in scene.sensors))

        index_local = _mujoco_tactile_points_to_urdf_local(
            _load_tactile_xml_points(XHAND_TACTILE_DIR / "index_collisions.xml"),
            finger="index",
            side="right",
        )
        thumb_right_local = _mujoco_tactile_points_to_urdf_local(
            _load_tactile_xml_points(XHAND_TACTILE_DIR / "thumb_collisions.xml"),
            finger="thumb",
            side="right",
        )
        thumb_left_local = _mujoco_tactile_points_to_urdf_local(
            _load_tactile_xml_points(XHAND_TACTILE_DIR / "thumb_collisions.xml"),
            finger="thumb",
            side="left",
        )
        self.assertGreater(float(index_local[:, 2].min()), 0.017)
        self.assertGreater(float(index_local[:, 2].max()), 0.041)
        self.assertGreater(float(thumb_right_local[:, 1].max()), 0.050)
        self.assertLess(float(thumb_left_local[:, 1].min()), -0.050)


if __name__ == "__main__":
    unittest.main()
