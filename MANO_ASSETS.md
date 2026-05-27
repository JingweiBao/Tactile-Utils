# MANO Assets

This project uses MANO only as a local asset. MANO model files are license-controlled and should not be committed to this repository.

The whole `assets/hands/` directory is local-only in this project. It may contain MANO files and vendor-provided dexterous-hand engineering files, so it is ignored by git.

## Official Download

Download MANO from the official website:

```text
https://mano.is.tue.mpg.de/
```

Steps:

1. Register or sign in on the MANO website.
2. Agree to the MANO license.
3. Download the MANO Models & Code package, normally named like `mano_v1_2.zip`.
4. Extract it locally under this project:

```text
assets/hands/mano/mano_v1_2/
```

The MANO website requires login for the download page, so this project does not provide an automatic download script.

## Required Files

Minimum required files for MANO-based shape alignment:

```text
assets/hands/mano/mano_v1_2/models/MANO_LEFT.pkl
assets/hands/mano/mano_v1_2/models/MANO_RIGHT.pkl
```

Required files for MANO UV-based projection:

```text
assets/hands/mano/MANO_UV_left.obj
assets/hands/mano/MANO_UV_right.obj
```

Optional helper note:

```text
assets/hands/mano/MANO_UV_directions.txt
```

The UV OBJ files are expected to contain MANO topology-compatible UV information. They are used as UV atlas references; fitted MANO meshes should copy/use the corresponding UV coordinates and face-UV indices.

## Not Required For This Project

These files may be present in the official package, but they are not required for the current tactile utilities:

```text
assets/hands/mano/mano_v1_2/models/SMPLH_male.pkl
assets/hands/mano/mano_v1_2/models/SMPLH_female.pkl
assets/hands/mano/mano_v1_2.zip
```

The zip can be deleted after extraction. The SMPLH files are large and should not be uploaded to GitHub unless a future module explicitly needs them and the license allows your use case.

## Expected Local Layout

```text
assets/hands/mano/
  MANO_UV_left.obj
  MANO_UV_right.obj
  MANO_UV_directions.txt
  mano_v1_2/
    models/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
    webuser/
      ...
```

## Quick Check

Run this from the project root:

```bash
test -f assets/hands/mano/mano_v1_2/models/MANO_LEFT.pkl
test -f assets/hands/mano/mano_v1_2/models/MANO_RIGHT.pkl
test -f assets/hands/mano/MANO_UV_left.obj
test -f assets/hands/mano/MANO_UV_right.obj
```

Or print what is missing:

```bash
for f in \
  assets/hands/mano/mano_v1_2/models/MANO_LEFT.pkl \
  assets/hands/mano/mano_v1_2/models/MANO_RIGHT.pkl \
  assets/hands/mano/MANO_UV_left.obj \
  assets/hands/mano/MANO_UV_right.obj
do
  [ -f "$f" ] || echo "missing: $f"
done
```

## GitHub Policy

Do not commit MANO/SMPLH model files or downloaded zip archives to GitHub. Keep them as local assets and document the expected paths instead.

Recommended `.gitignore` entries:

```text
assets/hands/
```
