# Online XHand Tactile To MANO Mapping

The online module consumes one runtime frame:

```text
current XHand qpos + current tactile values
  -> XHand semantic keypoints through URDF FK
  -> calibrated qpos-to-MANO retarget model, or analytic fallback
  -> posed fitted MANO mesh
  -> optional MANO-frame wrist/root transform
  -> offline taxel-to-MANO sparse / barycentric mapping
  -> vertex tactile values and taxel surface points
```

It reuses the offline reference assets and does not run per-frame optimization.

## Command

```bash
PYTHONPATH=src online-tactile-map map-xhand-tactile \
  --side right \
  --alignment-json results/offline_shape_alignment/<timestamp>_xhand_both_mano_beta_pose_fit.json \
  --projection-npz results/tactile_to_mano_projection/<timestamp>_xhand_both_graph_preserving_no_block_sensor_to_mano_weights_right.npz \
  --qpos-json current_qpos.json \
  --tactile-values current_tactile.npy \
  --retarget-model-npz results/online_tactile_mapping/<timestamp>_xhand_right_mano_retarget_model.npz \
  --mano-frame-transform-json optional_current_wrist_transform.json \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --results-root results
```

`--qpos-json` accepts either a direct joint mapping or an object with a `qpos` mapping. Missing joints fall back to the offline reference `robot_qpos`. `--tactile-values` accepts a `.npy` vector or JSON containing either a vector or a `tactile_values` key.

`--retarget-model-npz` is optional. When supplied, online pose generation uses a pre-calibrated fast regression model from XHand qpos to MANO pose. Without it, the module falls back to analytic keypoint-frame retargeting.

`--mano-frame-transform-json` is optional. It accepts a 4x4 transform, or an object with key `mano_frame_transform`, and applies it to the posed MANO vertices/taxel surface points after local qpos retargeting. This is the hook for current wrist/root alignment when the caller has an arm/end-effector pose. The real-data test script computes this transform from the relative XHand wrist pose as:

```text
mano_frame_transform = robot_to_mano * xhand_root_delta * inv(robot_to_mano)
```

## Retarget Calibration

The recommended pose path is:

```text
sample many XHand qpos
  -> solve pose-only MANO IK with fixed beta for each qpos
  -> fit a ridge qpos-to-MANO-pose regression model
  -> load that model during online tactile mapping
```

Run a calibration job:

```bash
PYTHONPATH=src online-tactile-map calibrate-retarget \
  --side right \
  --alignment-json results/offline_shape_alignment/<timestamp>_xhand_both_mano_beta_pose_fit.json \
  --projection-npz results/tactile_to_mano_projection/<timestamp>_xhand_both_graph_preserving_no_block_sensor_to_mano_weights_right.npz \
  --sample-count 512 \
  --iterations 160 \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --results-root results
```

The calibration command fixes MANO beta, optimizes only MANO pose against current XHand semantic keypoints and finger directions, then stores `qpos_names`, `pose_labels`, regression coefficients, calibration qpos samples, optimized poses, and IK error statistics in the model `.npz`.

## Outputs

Default outputs are written under:

```text
results/online_tactile_mapping/
```

The `.npz` file stores:

```text
mano_vertices
mano_faces
vertex_tactile_values
taxel_surface_points
mano_pose
mano_frame_transform
tactile_values
valid_vertex_mask
qpos_names
qpos_values
```

The JSON report records side, taxel / vertex counts, tactile statistics, pose norm statistics, merged qpos, the MANO-frame transform, and output paths. The PNG renders the current posed MANO mesh with projected taxel surface points colored by tactile value.

## Scope

This first online version assumes the runtime hand is the same XHand topology used offline. It reuses the fitted MANO beta, optional offline pose residual, taxel order, sparse vertex weights, and nearest-face barycentric mapping. `xhand_qpos_to_mano_pose` now returns a 16-joint MANO pose with a root/wrist entry followed by the 15 non-root joints; the root entry is the analytic palm-frame delta from reference to current qpos. It does not update MANO beta, re-project taxels, or model contact deformation.
