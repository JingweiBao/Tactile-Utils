from __future__ import annotations

import argparse
from pathlib import Path

from tactile_layout_3d_projection.config import load_sensor_mount_config
from tactile_utils_common.results import make_result_timestamp, module_result_path


MODULE_NAME = "tactile_layout_3d_projection"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tactile-utils",
        description="Visualize dexterous-hand tactile taxel locations in the hand URDF 3D frame.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    visualize_inspire = subparsers.add_parser(
        "visualize-inspire-3d",
        help="Render Inspire tactile mounting config points on the Inspire URDF mesh.",
    )
    visualize_inspire.add_argument("--mount-config", required=True)
    visualize_inspire.add_argument("--right-urdf", required=True)
    visualize_inspire.add_argument("--left-urdf", required=True)
    visualize_inspire.add_argument("--side", choices=["left", "right", "both"], default="both")
    visualize_inspire.add_argument("--view-mode", choices=["palm", "same-camera"], default="palm")
    _add_result_args(visualize_inspire)
    visualize_inspire.add_argument("--out")
    visualize_inspire.add_argument("--out-points")
    visualize_inspire.set_defaults(func=_cmd_visualize_inspire_3d)

    visualize_xhand = subparsers.add_parser(
        "visualize-xhand-3d",
        help="Render XHand official tactile XML points on the XHand URDF mesh.",
    )
    visualize_xhand.add_argument(
        "--xhand-root",
        default="assets/hands/xhand",
        help="Root directory containing Xhand-urdf and tactile sensor assets",
    )
    visualize_xhand.add_argument("--right-urdf")
    visualize_xhand.add_argument("--left-urdf")
    visualize_xhand.add_argument("--tactile-dir")
    visualize_xhand.add_argument("--side", choices=["left", "right", "both"], default="both")
    visualize_xhand.add_argument("--view-mode", choices=["palm", "same-camera"], default="same-camera")
    _add_result_args(visualize_xhand)
    visualize_xhand.add_argument("--out")
    visualize_xhand.add_argument("--out-points")
    visualize_xhand.set_defaults(func=_cmd_visualize_xhand_3d)

    args = parser.parse_args()
    args.func(args)


def _add_result_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-root",
        default="results",
        help=f"Root results directory; outputs are saved under <root>/{MODULE_NAME}/",
    )
    parser.add_argument(
        "--timestamp",
        help="Timestamp prefix for output filenames, for example 20260527_161500. Defaults to current local time.",
    )


def _cmd_visualize_inspire_3d(args: argparse.Namespace) -> None:
    from tactile_layout_3d_projection.inspire import (
        build_inspire_3d_scene,
        export_sensor_points_3d,
        render_inspire_3d_scene,
        render_inspire_3d_scenes,
    )

    config = load_sensor_mount_config(args.mount_config)
    urdfs = {"left": args.left_urdf, "right": args.right_urdf}
    sides = ["left", "right"] if args.side == "both" else [args.side]
    out_path, points_path = _resolve_result_paths(
        args,
        default_image_name=f"inspire_{args.side}_{args.view_mode}.png",
        default_points_name=f"inspire_{args.side}_sensor_points.pkl",
    )
    scenes = [build_inspire_3d_scene(config, urdfs[side], side=side) for side in sides]
    if len(scenes) == 1:
        render_inspire_3d_scene(scenes[0], out_path, view_mode=args.view_mode)
    else:
        render_inspire_3d_scenes(scenes, out_path, view_mode=args.view_mode)
    if points_path:
        export_sensor_points_3d(scenes, points_path)
    print(f"OK: wrote {out_path}")
    if points_path:
        print(f"OK: wrote {points_path}")


def _cmd_visualize_xhand_3d(args: argparse.Namespace) -> None:
    from tactile_layout_3d_projection.xhand import (
        build_xhand_3d_scene,
        export_xhand_sensor_points_3d,
        render_xhand_3d_scenes,
    )

    root = Path(args.xhand_root)
    right_urdf = args.right_urdf or root / "Xhand-urdf" / "xhand1_right(1)" / "urdf" / "xhand_right.urdf"
    left_urdf = args.left_urdf or root / "Xhand-urdf" / "xhand1_left(1)" / "urdf" / "xhand_left.urdf"
    tactile_dir = args.tactile_dir or root / "tactile sensor" / "xhand_tac"
    urdfs = {"left": left_urdf, "right": right_urdf}
    sides = ["left", "right"] if args.side == "both" else [args.side]
    out_path, points_path = _resolve_result_paths(
        args,
        default_image_name=f"xhand_{args.side}_{args.view_mode}.png",
        default_points_name=f"xhand_{args.side}_sensor_points.pkl",
    )
    scenes = [build_xhand_3d_scene(urdfs[side], tactile_dir, side=side) for side in sides]
    render_xhand_3d_scenes(scenes, out_path, view_mode=args.view_mode)
    if points_path:
        export_xhand_sensor_points_3d(scenes, points_path)
    print(f"OK: wrote {out_path}")
    if points_path:
        print(f"OK: wrote {points_path}")


def _resolve_result_paths(
    args: argparse.Namespace,
    *,
    default_image_name: str,
    default_points_name: str,
) -> tuple[Path, Path | None]:
    timestamp = args.timestamp or make_result_timestamp()
    image_path = module_result_path(
        MODULE_NAME,
        args.out or default_image_name,
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".png",
    )
    points_path = None
    if args.out_points:
        points_path = module_result_path(
            MODULE_NAME,
            args.out_points or default_points_name,
            results_root=args.results_root,
            timestamp=timestamp,
            default_suffix=".pkl",
        )
    return image_path, points_path


if __name__ == "__main__":
    main()
