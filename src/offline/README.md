# Offline Tactile To MANO Projection

This package bridges the two existing submodules:

- `sub_modules.tactile_layout_3d_projection`: reads XHand tactile taxel locations in the URDF/reference frame.
- `sub_modules.offline_shape_alignment`: provides `robot_to_mano`, fitted robot reference `qpos`, and fitted MANO OBJ meshes.

The first MVP projects each tactile taxel center onto the fitted MANO mesh and stores a sparse MANO-vertex weight matrix.

## Command

```bash
PYTHONPATH=src python -m offline.cli project-xhand-tactile \
  --side both \
  --alignment-json results/offline_shape_alignment/<timestamp>_xhand_both_mano_beta_pose_fit.json \
  --xhand-root assets/hands/xhand \
  --projection-mode graph-preserving-no-block \
  --results-root results
```

`graph-preserving-no-block` is the current default projection mode, so `--projection-mode` may be omitted when using the recommended path.

If the alignment JSON does not contain `outputs.obj`, pass the fitted MANO OBJ explicitly:

```bash
PYTHONPATH=src python -m offline.cli project-xhand-tactile \
  --side right \
  --alignment-json results/offline_shape_alignment/<timestamp>_xhand_both_mano_beta_pose_fit.json \
  --fitted-mano-obj results/offline_shape_alignment/<timestamp>_xhand_both_fitted_mano_beta_pose_right.obj
```

For `--side both`, `--fitted-mano-obj` may contain `{side}`.

## Outputs

Default outputs are written under:

```text
results/tactile_to_mano_projection/
```

The `.npz` file stores the sparse matrix in COO format:

```text
rows, cols, vals, shape
```

The intended matrix shape is:

```text
[MANO vertex count, tactile taxel count]
```

Each taxel column sums to 1 and usually contributes to the three vertices of the nearest MANO triangle through barycentric weights. The file also stores projected points, nearest face ids, barycentric coordinates, taxel metadata, and a valid MANO vertex mask. In `block-preserving` and `graph-preserving` modes it additionally stores `block_ids`, `block_local_uv`, `block_target_points`, `block_similarity_scale`, `block_fallback_reasons`, and `block_layout_preservation`. In `graph-preserving` mode it also stores `graph_knn_indices`, `graph_edge_length_error`, `graph_neighbor_mismatch`, `graph_angle_error_rad`, `graph_laplacian_error`, and `graph_fallback_reasons`.

This is a static reference-pose projection. It does not yet solve online robot pose retargeting or dynamic contact deformation.

Each run writes two PNG styles:

- `*_projection*.png`: full diagnostic view with original transformed taxel points, projected surface points, and connecting lines.
- `*_surface*.png`: MANO-only surface view with just the projected taxel points after projection.

## Quality Evaluation

The JSON report groups evaluation metrics into three categories:

- `surface_fitting`: MANO surface fitting quality, including nearest distance, global-nearest baseline distance, semantic distance delta, warning count, and warning ratio.
- `graph_preservation`: local graph preservation, including kNN edge-length error, neighbor mismatch, local angle error, and graph Laplacian error.
- `distribution_quality`: projected patch distribution, including nearest-neighbor spacing CV, collapse ratio under a 1 mm threshold, PCA convex-hull coverage area, PCA spread, occupied face count, and occupied vertex count.

These groups answer different questions. Lower distance is not automatically better if the patch collapses, and lower edge/Laplacian error is not automatically better if neighbor mismatch or angle error becomes worse. Based on the current XHand/MANO ablation, `graph-preserving-no-block` is the default candidate because it preserves better surface fitting, neighbor structure, angle structure, and distribution uniformity than the block-preserving branch.

## Projection Modes

- `global-nearest`: baseline mode. Each taxel uses the nearest triangle over the whole fitted MANO mesh.
- `semantic-distal`: conservative MVP mode. Each taxel infers its finger from the XHand sensor name / semantic region, then only searches the same MANO finger's distal/tip candidate faces. The JSON report keeps the global nearest distance for comparison and records fallback / warning counts.
- `semantic-normalized`: second-stage semantic mode. It computes each taxel's normalized coordinate in the XHand finger segment using the robot 21 keypoints, maps that coordinate to the same MANO finger segment, then projects to the same distal/tip candidate faces used by `semantic-distal`. If the normalized coordinate cannot be computed, it falls back to `semantic-distal`.
- `block-preserving`: third-stage semantic mode. It handles one `layout.sensor_slices` sensor block at a time, builds a block-local 2D coordinate system from the transformed XHand taxel points, aligns the local `u` axis to the same finger's DIP-to-tip direction, reconstructs the whole block around the MANO `semantic-normalized` target center, then snaps the reconstructed points to the same finger distal/tip candidate faces. Its fallback chain is `block-preserving -> semantic-normalized -> semantic-distal -> global-nearest`.
- `graph-preserving`: fourth-stage semantic mode. It starts from `block-preserving`, builds a kNN graph inside each sensor block, then performs a small surface-snap refinement that tries to keep source kNN edge lengths, neighbor overlap, local angle structure, and normalized graph Laplacian structure. Its fallback chain is `graph-preserving -> block-preserving -> semantic-normalized -> semantic-distal -> global-nearest`.
- `graph-preserving-no-block`: current default candidate. It disables the `block-preserving` reconstruction and applies the same kNN graph refinement directly on top of `semantic-normalized`, avoiding the block-level shape constraint that increased distance, neighbor mismatch, angle error, and distribution non-uniformity in the current ablation.

`block-preserving` is intentionally more conservative about tactile patch layout than nearest-distance minimization. Its nearest-distance metrics can be worse than `semantic-distal`, but the 120 taxels in each XHand finger sensor should remain a more continuous, regular patch on the MANO surface.
`graph-preserving` is even more conservative about local topology. It may keep the same surface points as `block-preserving` when the graph refinement would make the kNN preservation score worse; this rejection is recorded in `graph_fallback_reasons`.
