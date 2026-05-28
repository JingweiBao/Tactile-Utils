from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from offline.tactile_to_mano_projection import (
    build_xhand_tactile_layout,
    load_alignment_side_config,
    load_obj_mesh,
    project_tactile_layout_to_mano,
    projection_result_to_json,
    render_projected_surface_result,
    render_projection_result,
    save_projection_npz,
)
from sub_modules.offline_shape_alignment.xhand import default_xhand_urdf_path, infer_xhand_semantic_keypoints
from tactile_utils_common.results import make_result_timestamp, module_result_path


MODULE_NAME = "tactile_to_mano_projection"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="offline-tactile-projection",
        description="Precompute tactile sensor layout to fitted MANO mesh projection weights.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    project = subparsers.add_parser(
        "project-xhand-tactile",
        help="Project XHand tactile taxel centers onto a fitted MANO mesh.",
    )
    project.add_argument("--side", choices=["left", "right", "both"], default="right")
    project.add_argument("--xhand-root", default="assets/hands/xhand")
    project.add_argument("--tactile-dir")
    project.add_argument("--left-urdf")
    project.add_argument("--right-urdf")
    project.add_argument("--alignment-json", required=True)
    project.add_argument(
        "--projection-mode",
        choices=[
            "global-nearest",
            "semantic-distal",
            "semantic-normalized",
            "block-preserving",
            "graph-preserving",
            "graph-preserving-no-block",
        ],
        default="graph-preserving-no-block",
        help=(
            "Projection rule. semantic-distal restricts each taxel to the same MANO finger distal/tip region; "
            "semantic-normalized also maps finger-local coordinates before projection; block-preserving keeps "
            "each tactile sensor block's local layout before snapping to the MANO surface; graph-preserving "
            "adds kNN edge/adjacency/local-angle preservation. graph-preserving-no-block is the current default "
            "candidate and skips block-preserving."
        ),
    )
    project.add_argument(
        "--fitted-mano-obj",
        help="Override fitted MANO OBJ path. For --side both, use a string containing {side}.",
    )
    project.add_argument("--results-root", default="results")
    project.add_argument("--timestamp")
    project.add_argument("--out-json")
    project.add_argument("--out-npz")
    project.add_argument("--out-png")
    project.add_argument("--out-surface-png")
    project.add_argument("--out-mask")
    project.set_defaults(func=_cmd_project_xhand_tactile)

    args = parser.parse_args(argv)
    args.func(args)


def _cmd_project_xhand_tactile(args: argparse.Namespace) -> None:
    sides = ["left", "right"] if args.side == "both" else [args.side]
    timestamp = args.timestamp or make_result_timestamp()
    tactile_dir = Path(args.tactile_dir) if args.tactile_dir else Path(args.xhand_root) / "tactile sensor" / "xhand_tac"

    json_path = module_result_path(
        MODULE_NAME,
        args.out_json or f"xhand_{args.side}_{_mode_token(args.projection_mode)}_sensor_to_mano_projection.json",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".json",
    )

    payload = {
        "module": MODULE_NAME,
        "command": "project-xhand-tactile",
        "projection_mode": args.projection_mode,
        "alignment_json": str(args.alignment_json),
        "sides": {},
        "outputs": {
            "weights_npz": {},
            "visualization_png": {},
            "surface_png": {},
            "valid_vertex_mask_npy": {},
        },
    }

    for side in sides:
        alignment = load_alignment_side_config(
            args.alignment_json,
            side,
            fitted_mano_obj=_resolve_obj_override(args.fitted_mano_obj, side),
        )
        if alignment.fitted_mano_obj is None:
            raise ValueError(
                f"no fitted MANO OBJ found for side {side!r}; pass --fitted-mano-obj or use an alignment JSON with outputs.obj"
            )
        urdf_path = _xhand_urdf_for_side(args, side)
        layout = build_xhand_tactile_layout(
            urdf_path,
            tactile_dir,
            side,
            qpos=alignment.robot_qpos,
        )
        robot_keypoints = None
        if args.projection_mode in {
            "semantic-normalized",
            "block-preserving",
            "graph-preserving",
            "graph-preserving-no-block",
        }:
            robot_keypoints = infer_xhand_semantic_keypoints(urdf_path, side, qpos=alignment.robot_qpos).point_by_label()
        mesh = load_obj_mesh(alignment.fitted_mano_obj)
        result = project_tactile_layout_to_mano(
            layout,
            mesh,
            alignment.robot_to_mano,
            projection_mode=args.projection_mode,
            mano_keypoints=alignment.mano_keypoints,
            robot_keypoints=robot_keypoints,
            metadata=alignment.metadata,
        )

        npz_path = _side_result_path(
            args.out_npz or f"xhand_{args.side}_{_mode_token(args.projection_mode)}_sensor_to_mano_weights.npz",
            side,
            side_count=len(sides),
            results_root=args.results_root,
            timestamp=timestamp,
            default_suffix=".npz",
        )
        png_path = _side_result_path(
            args.out_png or f"xhand_{args.side}_{_mode_token(args.projection_mode)}_sensor_to_mano_projection.png",
            side,
            side_count=len(sides),
            results_root=args.results_root,
            timestamp=timestamp,
            default_suffix=".png",
        )
        surface_png_path = _side_result_path(
            args.out_surface_png or f"xhand_{args.side}_{_mode_token(args.projection_mode)}_sensor_to_mano_surface.png",
            side,
            side_count=len(sides),
            results_root=args.results_root,
            timestamp=timestamp,
            default_suffix=".png",
        )
        mask_path = _side_result_path(
            args.out_mask or f"xhand_{args.side}_{_mode_token(args.projection_mode)}_valid_vertex_mask.npy",
            side,
            side_count=len(sides),
            results_root=args.results_root,
            timestamp=timestamp,
            default_suffix=".npy",
        )

        save_projection_npz(result, npz_path)
        np.save(mask_path, result.valid_vertex_mask.astype(np.bool_))
        render_projection_result(result, png_path)
        render_projected_surface_result(result, surface_png_path)

        payload["sides"][side] = projection_result_to_json(result)
        payload["outputs"]["weights_npz"][side] = str(npz_path)
        payload["outputs"]["visualization_png"][side] = str(png_path)
        payload["outputs"]["surface_png"][side] = str(surface_png_path)
        payload["outputs"]["valid_vertex_mask_npy"][side] = str(mask_path)
        print(f"OK: wrote {side} weights {npz_path}")
        print(f"OK: wrote {side} mask {mask_path}")
        print(f"OK: wrote {side} visualization {png_path}")
        print(f"OK: wrote {side} surface visualization {surface_png_path}")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"OK: wrote {json_path}")


def _resolve_obj_override(value: str | None, side: str) -> Path | None:
    if value is None:
        return None
    return Path(value.format(side=side))


def _mode_token(mode: str) -> str:
    return mode.replace("-", "_")


def _xhand_urdf_for_side(args: argparse.Namespace, side: str) -> Path:
    if side == "left" and args.left_urdf:
        return Path(args.left_urdf)
    if side == "right" and args.right_urdf:
        return Path(args.right_urdf)
    return default_xhand_urdf_path(args.xhand_root, side)


def _side_result_path(
    filename: str,
    side: str,
    *,
    side_count: int,
    results_root: str | Path,
    timestamp: str,
    default_suffix: str,
) -> Path:
    path = Path(filename)
    if side_count > 1:
        filename = f"{path.stem}_{side}{path.suffix or default_suffix}"
    return module_result_path(
        MODULE_NAME,
        filename,
        results_root=results_root,
        timestamp=timestamp,
        default_suffix=default_suffix,
    )


if __name__ == "__main__":
    main()
