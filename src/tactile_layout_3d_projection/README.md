# 3D Tactile Layout Projection

This module aligns tactile taxel locations to a dexterous hand's own 3D URDF frame and renders them against the full hand mesh.

Code in this folder is intentionally self-contained:

- `cli.py`: module command-line entry points.
- `config.py`: Inspire-style external mounting config loader.
- `geometry.py`: rigid transforms and taxel-grid generation.
- `urdf.py`: URDF link/joint parsing and forward kinematics.
- `inspire.py`: Inspire mounting config to 3D scene.
- `xhand.py`: XHand official tactile XML to 3D scene.

Shared resources such as URDFs, meshes, tactile XML files, and configs remain in the project-level `assets/` and `configs/` folders. Generated outputs are grouped by module under:

```text
results/tactile_layout_3d_projection/
```

Output filenames are prefixed with local time in `YYYYMMDD_HHMMSS` format so repeated runs stay sortable, for example:

```text
results/tactile_layout_3d_projection/20260527_161500_xhand_both_same-camera.png
```
