from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from offline.tactile_to_mano_projection import (
    PROJECTION_MODES,
    TactileLayout,
    apply_sensor_values_to_vertices,
    build_mano_semantic_regions,
    load_obj_mesh,
    normalize_finger_name,
    project_points_to_mesh,
    project_tactile_layout_to_mano,
    projection_result_to_json,
    render_projected_surface_result,
    save_projection_npz,
)
from sub_modules.offline_shape_alignment.types import Mesh


class OfflineTactileToManoProjectionTest(unittest.TestCase):
    def test_normalize_finger_name_maps_mid_to_middle(self) -> None:
        self.assertEqual(normalize_finger_name("right_mid_tactile_sensor"), "middle")
        self.assertEqual(normalize_finger_name("left_middle"), "middle")
        self.assertEqual(normalize_finger_name("right_index_tip"), "index")

    def test_project_points_to_mesh_returns_barycentric_weights(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        )
        points = np.asarray([[0.25, 0.25, 0.2]], dtype=np.float64)

        face_indices, barycentric, projected, distances = project_points_to_mesh(points, mesh)

        self.assertEqual(face_indices.tolist(), [0])
        self.assertTrue(np.allclose(barycentric.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(barycentric[0], [0.5, 0.25, 0.25]))
        self.assertTrue(np.allclose(projected[0], [0.25, 0.25, 0.0]))
        self.assertAlmostEqual(float(distances[0]), 0.2)

    def test_projection_result_sparse_columns_sum_to_one(self) -> None:
        layout = TactileLayout(
            side="right",
            points_robot=np.asarray(
                [
                    [0.25, 0.25, 0.1],
                    [0.75, 0.25, 0.1],
                ],
                dtype=np.float64,
            ),
            sensor_names=("right_index_tactile_sensor", "right_index_tactile_sensor"),
            semantic_regions=("right_index", "right_index"),
            taxel_indices=np.asarray([[0, 0], [1, 0]], dtype=np.int64),
            sensor_slices=(
                {
                    "name": "right_index_tactile_sensor",
                    "semantic_region": "right_index",
                    "link_name": "right_hand_index_rota_link2",
                    "start": 0,
                    "count": 2,
                },
            ),
            metadata={"hand": "xhand"},
        )
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )

        result = project_tactile_layout_to_mano(layout, mesh, np.eye(4, dtype=np.float64))
        column_sums = np.zeros(result.shape[1], dtype=np.float64)
        np.add.at(column_sums, result.cols, result.vals)

        self.assertEqual(result.shape, (4, 2))
        self.assertTrue(np.allclose(column_sums, 1.0))
        self.assertEqual(result.valid_vertex_mask.shape, (4,))
        self.assertGreaterEqual(int(result.valid_vertex_mask.sum()), 3)

        sensor_values = np.asarray([2.0, 4.0], dtype=np.float64)
        vertex_values = apply_sensor_values_to_vertices(
            result.rows,
            result.cols,
            result.vals,
            result.shape,
            sensor_values,
        )
        self.assertEqual(vertex_values.shape, (4,))
        self.assertTrue(np.isfinite(vertex_values).all())

        payload = projection_result_to_json(result)
        self.assertIn("quality_evaluation", payload)
        quality = payload["quality_evaluation"]
        self.assertIn("surface_fitting", quality)
        self.assertIn("graph_preservation", quality)
        self.assertIn("distribution_quality", quality)
        self.assertIn("nearest_neighbor_distance_mm", quality["distribution_quality"])
        self.assertIn("collapse_ratio", quality["distribution_quality"])
        self.assertIn("coverage_area_mm2", quality["distribution_quality"])

    def test_obj_loader_and_npz_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "mesh.obj"
            obj_path.write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 0 1 0",
                        "f 1 2 3",
                    ]
                ),
                encoding="utf-8",
            )
            mesh = load_obj_mesh(obj_path)
            layout = TactileLayout(
                side="right",
                points_robot=np.asarray([[0.2, 0.2, 0.0]], dtype=np.float64),
                sensor_names=("right_index_tactile_sensor",),
                semantic_regions=("right_index",),
                taxel_indices=np.asarray([[0, 0]], dtype=np.int64),
                sensor_slices=(
                    {
                        "name": "right_index_tactile_sensor",
                        "semantic_region": "right_index",
                        "link_name": "right_hand_index_rota_link2",
                        "start": 0,
                        "count": 1,
                    },
                ),
                metadata={},
            )
            result = project_tactile_layout_to_mano(layout, mesh, np.eye(4, dtype=np.float64))
            npz_path = Path(tmpdir) / "weights.npz"
            surface_png_path = Path(tmpdir) / "surface.png"
            save_projection_npz(result, npz_path)
            render_projected_surface_result(result, surface_png_path, width=220, height=180)

            self.assertEqual(mesh.vertices.shape, (3, 3))
            self.assertEqual(mesh.faces.shape, (1, 3))
            self.assertTrue(npz_path.exists())
            self.assertTrue(surface_png_path.exists())
            self.assertGreater(surface_png_path.stat().st_size, 0)
            with np.load(npz_path) as payload:
                self.assertEqual(payload["shape"].tolist(), [3, 1])
                self.assertEqual(payload["rows"].shape, (3,))
                self.assertEqual(payload["cols"].shape, (3,))
                self.assertTrue(np.allclose(payload["vals"].sum(), 1.0))

    def test_semantic_distal_projection_restricts_to_target_finger(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [-0.1, 2.5, 0.0],
                    [0.1, 2.5, 0.0],
                    [0.0, 3.0, 0.0],
                    [9.9, 2.5, 0.0],
                    [10.1, 2.5, 0.0],
                    [10.0, 3.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        )
        layout = TactileLayout(
            side="right",
            points_robot=np.asarray([[10.0, 2.7, 0.0]], dtype=np.float64),
            sensor_names=("right_index_tactile_sensor",),
            semantic_regions=("right_index",),
            taxel_indices=np.asarray([[0, 0]], dtype=np.int64),
            sensor_slices=(
                {
                    "name": "right_index_tactile_sensor",
                    "semantic_region": "right_index",
                    "link_name": "right_hand_index_rota_link2",
                    "start": 0,
                    "count": 1,
                },
            ),
            metadata={},
        )

        result = project_tactile_layout_to_mano(
            layout,
            mesh,
            np.eye(4, dtype=np.float64),
            projection_mode="semantic-distal",
            mano_keypoints=_toy_mano_keypoints(),
        )

        self.assertEqual(result.projection_mode, "semantic-distal")
        self.assertEqual(result.global_nearest_face_indices.tolist(), [1])
        self.assertEqual(result.nearest_face_indices.tolist(), [0])
        self.assertEqual(result.target_fingers, ("index",))
        self.assertEqual(result.fallback_reasons, ("none",))
        self.assertTrue(result.semantic_warning_mask[0])

    def test_semantic_distal_projection_falls_back_for_unknown_finger(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        )
        layout = TactileLayout(
            side="right",
            points_robot=np.asarray([[0.1, 0.1, 0.0]], dtype=np.float64),
            sensor_names=("right_unknown_sensor",),
            semantic_regions=("right_unknown",),
            taxel_indices=np.asarray([[0, 0]], dtype=np.int64),
            sensor_slices=(
                {
                    "name": "right_unknown_sensor",
                    "semantic_region": "right_unknown",
                    "link_name": "right_unknown_link",
                    "start": 0,
                    "count": 1,
                },
            ),
            metadata={},
        )

        result = project_tactile_layout_to_mano(
            layout,
            mesh,
            np.eye(4, dtype=np.float64),
            projection_mode="semantic-distal",
            mano_keypoints=_toy_mano_keypoints(),
        )

        self.assertEqual(result.fallback_reasons, ("unknown_finger_to_global",))
        self.assertEqual(result.nearest_face_indices.tolist(), result.global_nearest_face_indices.tolist())
        self.assertTrue(np.allclose(result.barycentric.sum(axis=1), 1.0))

    def test_semantic_normalized_maps_finger_local_coordinate_before_projection(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [-0.1, 4.2, 0.0],
                    [0.1, 4.2, 0.0],
                    [0.0, 4.3, 0.0],
                    [-0.1, 5.2, 0.0],
                    [0.1, 5.2, 0.0],
                    [0.0, 5.3, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        )
        layout = TactileLayout(
            side="right",
            points_robot=np.asarray([[0.0, 2.55, 0.0]], dtype=np.float64),
            sensor_names=("right_index_tactile_sensor",),
            semantic_regions=("right_index",),
            taxel_indices=np.asarray([[0, 0]], dtype=np.int64),
            sensor_slices=(
                {
                    "name": "right_index_tactile_sensor",
                    "semantic_region": "right_index",
                    "link_name": "right_hand_index_rota_link2",
                    "start": 0,
                    "count": 1,
                },
            ),
            metadata={},
        )
        robot_keypoints = _toy_mano_keypoints()
        mano_keypoints = _toy_mano_keypoints(index_length=6.0)

        distal = project_tactile_layout_to_mano(
            layout,
            mesh,
            np.eye(4, dtype=np.float64),
            projection_mode="semantic-distal",
            mano_keypoints=mano_keypoints,
        )
        normalized = project_tactile_layout_to_mano(
            layout,
            mesh,
            np.eye(4, dtype=np.float64),
            projection_mode="semantic-normalized",
            mano_keypoints=mano_keypoints,
            robot_keypoints=robot_keypoints,
        )

        self.assertEqual(distal.nearest_face_indices.tolist(), [0])
        self.assertEqual(normalized.nearest_face_indices.tolist(), [1])
        self.assertEqual(normalized.fallback_reasons, ("none",))
        self.assertTrue(np.isfinite(normalized.normalized_coordinates).all())
        self.assertGreater(float(normalized.semantic_target_points[0, 1]), 5.0)

    def test_block_preserving_keeps_local_block_distances(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [-0.5, 4.7, 0.0],
                    [0.5, 4.7, 0.0],
                    [0.5, 5.7, 0.0],
                    [-0.5, 5.7, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        points = np.asarray(
            [
                [-0.04, 2.50, 0.0],
                [0.04, 2.50, 0.0],
                [-0.04, 2.60, 0.0],
                [0.04, 2.60, 0.0],
            ],
            dtype=np.float64,
        )
        layout = TactileLayout(
            side="right",
            points_robot=points,
            sensor_names=tuple(["right_index_tactile_sensor"] * points.shape[0]),
            semantic_regions=tuple(["right_index"] * points.shape[0]),
            taxel_indices=np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64),
            sensor_slices=(
                {
                    "name": "right_index_tactile_sensor",
                    "semantic_region": "right_index",
                    "link_name": "right_hand_index_rota_link2",
                    "start": 0,
                    "count": int(points.shape[0]),
                },
            ),
            metadata={},
        )

        result = project_tactile_layout_to_mano(
            layout,
            mesh,
            np.eye(4, dtype=np.float64),
            projection_mode="block-preserving",
            mano_keypoints=_toy_mano_keypoints(index_length=6.0),
            robot_keypoints=_toy_mano_keypoints(index_length=3.0),
        )

        column_sums = np.zeros(result.shape[1], dtype=np.float64)
        np.add.at(column_sums, result.cols, result.vals)
        expected_distances = _pairwise_distances(result.block_local_uv * result.block_similarity_scale[:, None])
        target_distances = _pairwise_distances(result.block_target_points)

        self.assertEqual(result.projection_mode, "block-preserving")
        self.assertEqual(set(result.block_fallback_reasons), {"none"})
        self.assertTrue(np.allclose(column_sums, 1.0))
        self.assertTrue(np.isfinite(result.block_local_uv).all())
        self.assertTrue(np.isfinite(result.block_similarity_scale).all())
        self.assertTrue(np.all(np.isin(result.nearest_face_indices, [0, 1])))
        self.assertTrue(np.allclose(expected_distances, target_distances))
        self.assertLess(float(np.nanmax(result.block_layout_preservation)), 1e-9)

    def test_block_preserving_falls_back_for_too_small_block(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [-0.5, 4.7, 0.0],
                    [0.5, 4.7, 0.0],
                    [0.5, 5.7, 0.0],
                    [-0.5, 5.7, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        points = np.asarray([[-0.04, 2.50, 0.0], [0.04, 2.50, 0.0]], dtype=np.float64)
        layout = TactileLayout(
            side="right",
            points_robot=points,
            sensor_names=("right_index_tactile_sensor", "right_index_tactile_sensor"),
            semantic_regions=("right_index", "right_index"),
            taxel_indices=np.asarray([[0, 0], [1, 0]], dtype=np.int64),
            sensor_slices=(
                {
                    "name": "right_index_tactile_sensor",
                    "semantic_region": "right_index",
                    "link_name": "right_hand_index_rota_link2",
                    "start": 0,
                    "count": int(points.shape[0]),
                },
            ),
            metadata={},
        )

        result = project_tactile_layout_to_mano(
            layout,
            mesh,
            np.eye(4, dtype=np.float64),
            projection_mode="block-preserving",
            mano_keypoints=_toy_mano_keypoints(index_length=6.0),
            robot_keypoints=_toy_mano_keypoints(index_length=3.0),
        )

        column_sums = np.zeros(result.shape[1], dtype=np.float64)
        np.add.at(column_sums, result.cols, result.vals)

        self.assertEqual(set(result.block_fallback_reasons), {"block_too_small_to_semantic_normalized"})
        self.assertTrue(np.isnan(result.block_local_uv).all())
        self.assertTrue(np.allclose(column_sums, 1.0))
        self.assertEqual(result.fallback_reasons, ("none", "none"))

    def test_graph_preserving_reports_knn_structure(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [-0.5, 4.6, 0.0],
                    [0.5, 4.6, 0.0],
                    [0.5, 5.9, 0.0],
                    [-0.5, 5.9, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        points = np.asarray(
            [
                [-0.04, 2.45, 0.0],
                [0.04, 2.45, 0.0],
                [-0.04, 2.55, 0.0],
                [0.04, 2.55, 0.0],
                [-0.04, 2.65, 0.0],
                [0.04, 2.65, 0.0],
            ],
            dtype=np.float64,
        )
        layout = TactileLayout(
            side="right",
            points_robot=points,
            sensor_names=tuple(["right_index_tactile_sensor"] * points.shape[0]),
            semantic_regions=tuple(["right_index"] * points.shape[0]),
            taxel_indices=np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]], dtype=np.int64),
            sensor_slices=(
                {
                    "name": "right_index_tactile_sensor",
                    "semantic_region": "right_index",
                    "link_name": "right_hand_index_rota_link2",
                    "start": 0,
                    "count": int(points.shape[0]),
                },
            ),
            metadata={},
        )

        result = project_tactile_layout_to_mano(
            layout,
            mesh,
            np.eye(4, dtype=np.float64),
            projection_mode="graph-preserving",
            mano_keypoints=_toy_mano_keypoints(index_length=6.0),
            robot_keypoints=_toy_mano_keypoints(index_length=3.0),
        )

        column_sums = np.zeros(result.shape[1], dtype=np.float64)
        np.add.at(column_sums, result.cols, result.vals)

        self.assertIn("graph-preserving", PROJECTION_MODES)
        self.assertEqual(result.projection_mode, "graph-preserving")
        self.assertEqual(set(result.graph_fallback_reasons), {"none"})
        self.assertTrue(np.allclose(column_sums, 1.0))
        self.assertEqual(result.graph_knn_indices.shape, (points.shape[0], 4))
        self.assertTrue((result.graph_knn_indices >= 0).all())
        self.assertLess(float(np.nanmax(result.graph_edge_length_error)), 1e-9)
        self.assertLess(float(np.nanmax(result.graph_neighbor_mismatch)), 1e-9)
        self.assertLess(float(np.nanmax(result.graph_angle_error_rad)), 1e-9)

    def test_graph_preserving_no_block_skips_block_reconstruction(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [-0.5, 4.6, 0.0],
                    [0.5, 4.6, 0.0],
                    [0.5, 5.9, 0.0],
                    [-0.5, 5.9, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        points = np.asarray(
            [
                [-0.04, 2.45, 0.0],
                [0.04, 2.45, 0.0],
                [-0.04, 2.55, 0.0],
                [0.04, 2.55, 0.0],
                [-0.04, 2.65, 0.0],
                [0.04, 2.65, 0.0],
            ],
            dtype=np.float64,
        )
        layout = TactileLayout(
            side="right",
            points_robot=points,
            sensor_names=tuple(["right_index_tactile_sensor"] * points.shape[0]),
            semantic_regions=tuple(["right_index"] * points.shape[0]),
            taxel_indices=np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]], dtype=np.int64),
            sensor_slices=(
                {
                    "name": "right_index_tactile_sensor",
                    "semantic_region": "right_index",
                    "link_name": "right_hand_index_rota_link2",
                    "start": 0,
                    "count": int(points.shape[0]),
                },
            ),
            metadata={},
        )

        result = project_tactile_layout_to_mano(
            layout,
            mesh,
            np.eye(4, dtype=np.float64),
            projection_mode="graph-preserving-no-block",
            mano_keypoints=_toy_mano_keypoints(index_length=6.0),
            robot_keypoints=_toy_mano_keypoints(index_length=3.0),
        )

        column_sums = np.zeros(result.shape[1], dtype=np.float64)
        np.add.at(column_sums, result.cols, result.vals)

        self.assertEqual(result.projection_mode, "graph-preserving-no-block")
        self.assertEqual(set(result.block_fallback_reasons), {"block_preserving_disabled"})
        self.assertEqual(set(result.graph_fallback_reasons), {"none"})
        self.assertTrue(np.allclose(column_sums, 1.0))
        self.assertTrue((result.graph_knn_indices >= 0).all())
        self.assertTrue(np.isfinite(result.graph_edge_length_error).all())

    def test_graph_preserving_falls_back_when_block_falls_back(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [-0.5, 4.7, 0.0],
                    [0.5, 4.7, 0.0],
                    [0.5, 5.7, 0.0],
                    [-0.5, 5.7, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        points = np.asarray([[-0.04, 2.50, 0.0], [0.04, 2.50, 0.0]], dtype=np.float64)
        layout = TactileLayout(
            side="right",
            points_robot=points,
            sensor_names=("right_index_tactile_sensor", "right_index_tactile_sensor"),
            semantic_regions=("right_index", "right_index"),
            taxel_indices=np.asarray([[0, 0], [1, 0]], dtype=np.int64),
            sensor_slices=(
                {
                    "name": "right_index_tactile_sensor",
                    "semantic_region": "right_index",
                    "link_name": "right_hand_index_rota_link2",
                    "start": 0,
                    "count": int(points.shape[0]),
                },
            ),
            metadata={},
        )

        result = project_tactile_layout_to_mano(
            layout,
            mesh,
            np.eye(4, dtype=np.float64),
            projection_mode="graph-preserving",
            mano_keypoints=_toy_mano_keypoints(index_length=6.0),
            robot_keypoints=_toy_mano_keypoints(index_length=3.0),
        )

        self.assertEqual(set(result.block_fallback_reasons), {"block_too_small_to_semantic_normalized"})
        self.assertEqual(set(result.graph_fallback_reasons), {"graph_block_fallback_to_block_preserving"})
        self.assertTrue((result.graph_knn_indices == -1).all())

    def test_mano_semantic_regions_build_distal_faces(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [-0.1, 2.5, 0.0],
                    [0.1, 2.5, 0.0],
                    [0.0, 3.0, 0.0],
                    [9.9, 2.5, 0.0],
                    [10.1, 2.5, 0.0],
                    [10.0, 3.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        )

        regions = build_mano_semantic_regions(mesh, _toy_mano_keypoints())

        self.assertEqual(regions.distal_faces["index"].tolist(), [0])
        self.assertEqual(regions.distal_faces["middle"].tolist(), [1])


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    pair_i, pair_j = np.triu_indices(points.shape[0], k=1)
    return np.linalg.norm(points[pair_i] - points[pair_j], axis=1)


def _toy_mano_keypoints(index_length: float = 3.0) -> dict[str, np.ndarray]:
    def finger(x: float, length: float = 3.0) -> dict[str, np.ndarray]:
        return {
            "mcp": np.asarray([x, 0.0, 0.0], dtype=np.float64),
            "pip": np.asarray([x, length / 3.0, 0.0], dtype=np.float64),
            "dip": np.asarray([x, 2.0 * length / 3.0, 0.0], dtype=np.float64),
            "tip": np.asarray([x, length, 0.0], dtype=np.float64),
        }

    xs = {
        "thumb": -10.0,
        "index": 0.0,
        "middle": 10.0,
        "ring": 20.0,
        "pinky": 30.0,
    }
    keypoints = {"wrist": np.asarray([0.0, -1.0, 0.0], dtype=np.float64)}
    for name, x in xs.items():
        points = finger(x, index_length if name == "index" else 3.0)
        keypoints[f"{name}_mcp"] = points["mcp"]
        keypoints[f"{name}_pip"] = points["pip"]
        keypoints[f"{name}_dip"] = points["dip"]
        keypoints[f"{name}_tip"] = points["tip"]
    return keypoints


if __name__ == "__main__":
    unittest.main()
