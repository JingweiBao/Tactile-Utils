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
PYTHONPATH=src /home/jingwei/miniconda3/envs/forcevla/bin/python -m unittest discover -s tests -v
```
