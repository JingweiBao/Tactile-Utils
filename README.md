# tactile_utils

`tactile_utils` is a collection of small utilities for tactile-sensor data, calibration, visualization, and conversion.

The current implemented module is directly under `src/`:

```text
src/tactile_layout_3d_projection/
```

It visualizes dexterous-hand tactile sensor layouts in the hand's own 3D URDF frame.

Project-level resources stay at the repository root so future utilities can share them:

```text
assets/    local assets and shared docs
configs/   shared configuration files
results/   generated outputs, grouped by module
tests/     regression tests
```

`assets/hands/` is intentionally local-only and is ignored by git. It may contain MANO files and vendor-provided dexterous-hand engineering files such as URDFs, meshes, tactile XMLs, or calibration examples.

For MANO files, see [MANO_ASSETS.md](MANO_ASSETS.md) for download instructions and required local paths. For dexterous-hand engineering files, ask the corresponding hand vendor or hardware provider for the official package; this repository only documents the expected local layout.

## Environment

This project uses the Conda environment named `tactile_utils`.

Activate it before running project commands:

```bash
conda activate tactile_utils
```

Install future dependencies into this environment through Conda whenever possible:

```bash
conda install -n tactile_utils <package>
```

If a package is unavailable from Conda channels, install it with pip from inside the activated `tactile_utils` environment.

## Module: Offline Shape Alignment Diagnostics

The first `offline_shape_alignment` implementation is diagnostic-only. It does not optimize MANO shape parameters yet.

It currently supports XHand reference-pose diagnostics against MANO:

- Infer 21 XHand semantic keypoints from URDF joint/link names and FK.
- Generate 21 MANO semantic keypoints from the MANO reference model.
- Estimate a `robot_to_mano` similarity transform.
- Save a JSON report and a PNG visual comparison under `results/offline_shape_alignment/`.
- Optionally fit a conservative XHand reference `qpos` when the semantic keypoints match but the finger-base spread differs from MANO.

```bash
offline-shape-alignment diagnose-xhand-reference \
  --side right \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --out-json xhand_right_diagnostic.json \
  --out-png xhand_right_diagnostic.png
```

To diagnose with a fitted XHand reference pose, run:

```bash
offline-shape-alignment fit-xhand-reference-pose \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --out-json xhand_both_fit_diagnostic.json \
  --out-png xhand_both_fit_diagnostic.png \
  --out-pose-json xhand_both_robot_reference_pose.json
```

This writes the fitted robot `qpos`, baseline metrics, optimized metrics, and PNG comparison. It applies the fitted pose during diagnosis only; it does not rewrite URDF, mesh, or MANO assets.

The beta-only MANO shape optimization step requires PyTorch in the same Conda environment:

```bash
conda install -n tactile_utils pytorch cpuonly -c pytorch
```

If the CPU PyTorch package is installed against an incompatible MKL runtime, keep MKL below 2025:

```bash
conda install -n tactile_utils "mkl<2025"
```

Run beta-only shape fitting with:

```bash
offline-shape-alignment fit-mano-shape \
  --side right \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --iterations 1000 \
  --robot-surface-points 2048 \
  --mano-surface-points 2048 \
  --out-json xhand_right_mano_beta_fit.json \
  --out-png xhand_right_mano_beta_fit.png \
  --out-loss-png xhand_right_mano_beta_loss.png \
  --out-obj xhand_right_fitted_mano.obj
```

This optimizes MANO `beta` only. MANO pose remains the template / zero pose, and XHand is first aligned into the MANO frame using the diagnosed `robot_to_mano` transform.

After validating beta-only results, run the strongly regularized beta + pose residual MVP with:

```bash
offline-shape-alignment fit-mano-shape-pose \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --beta-init-iterations 400 \
  --iterations 600 \
  --robot-surface-points 2048 \
  --mano-surface-points 2048 \
  --pose-limit-rad 0.25 \
  --pose-l2-weight 1e-3 \
  --out-json xhand_both_mano_beta_pose_fit.json \
  --out-png xhand_both_mano_beta_pose_fit.png \
  --out-loss-png xhand_both_mano_beta_pose_loss.png \
  --out-obj xhand_both_fitted_mano_beta_pose.obj
```

This still keeps the MANO root fixed and constrains each non-root MANO joint residual through `pose_limit_rad * tanh(raw_pose)`. It is meant to test whether small MANO pose residuals improve alignment without letting pose absorb all robot-vs-human morphology differences.

## Module: 3D Tactile Layout Projection

Scope:

- Read the hand URDF and render the full hand mesh as a light gray reference model.
- Read tactile sensor positions from the best available hand-specific source.
- Transform tactile taxel centers into the same 3D frame as the URDF mesh.
- Save a PNG visualization and, optionally, a pickle file with the transformed 3D points.

### XHand

XHand tactile positions come from the official tactile simulation XML under:

```text
assets/hands/xhand/tactile sensor/xhand_tac/*_collisions.xml
```

These XHand files are not included in the repository. Prepare them locally under `assets/hands/xhand/`.

Each finger has 120 `class="pad"` geoms. These local MuJoCo tactile coordinates are converted into the corresponding XHand URDF distal-link frame before rendering.

```bash
tactile-utils visualize-xhand-3d \
  --xhand-root assets/hands/xhand \
  --side both \
  --view-mode palm \
  --out xhand_sensor_points_3d_palm_view.png \
  --out-points xhand_sensor_points_3d.pkl
```

### Inspire Hand

The local Inspire URDFs do not define tactile sensor frames. Inspire tactile points therefore come from the editable mounting seed:

```text
configs/inspire_hand_sensor_mount.yaml
```

This file defines each sensor's `parent_link`, `xyz`, `rpy`, `size_mm`, and `grid_shape`. It is a calibration seed, not hardware-ground-truth extrinsics.

The Inspire URDF and mesh files are not included in the repository. Prepare the vendor-provided hand engineering files locally under `assets/hands/inspire_hand_dexsuite/`.

```bash
tactile-utils visualize-inspire-3d \
  --mount-config configs/inspire_hand_sensor_mount.yaml \
  --right-urdf assets/hands/inspire_hand_dexsuite/inspire_hand_right.urdf \
  --left-urdf assets/hands/inspire_hand_dexsuite/inspire_hand_left.urdf \
  --side both \
  --view-mode palm \
  --out inspire_sensor_points_3d_palm_view.png \
  --out-points inspire_sensor_points_3d.pkl
```

### Outputs

CLI outputs are normalized into a module-specific results folder:

```text
results/tactile_layout_3d_projection/<YYYYMMDD_HHMMSS>_<filename>
```

For reproducible names, pass `--timestamp`, for example:

```bash
tactile-utils visualize-xhand-3d \
  --xhand-root assets/hands/xhand \
  --timestamp 20260527_161500 \
  --out xhand_both.png
```

The PNG output shows:

- Light gray translucent hand mesh from URDF collision geometry.
- Blue tactile taxel centers for the left hand.
- Orange tactile taxel centers for the right hand.

The optional point export is a Python pickle file containing one dictionary per side, then one entry per sensor:

```text
side -> sensor_name -> {
  side,
  semantic_region,
  grid_shape,
  taxel_indices,
  points
}
```

`points` are `float32` 3D coordinates in the rendered hand frame.

## Tests

```bash
conda activate tactile_utils
PYTHONPATH=src python -m unittest discover -s tests -v
```
