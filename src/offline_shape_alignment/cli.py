from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline_shape_alignment.alignment import diagnose_alignment
from offline_shape_alignment.mano import load_mano_reference
from offline_shape_alignment.reference_pose import fit_xhand_reference_pose, reference_pose_fit_to_json
from offline_shape_alignment.render import render_alignment_report, render_alignment_reports, render_loss_histories
from offline_shape_alignment.shape_optimization import (
    PoseShapeOptimizationConfig,
    ShapeOptimizationConfig,
    fit_mano_beta_to_xhand,
    fit_mano_beta_pose_to_xhand,
    pose_shape_optimization_result_to_json,
    shape_optimization_result_to_json,
    write_obj,
)
from offline_shape_alignment.xhand import default_xhand_urdf_path, load_xhand_reference
from tactile_utils_common.results import make_result_timestamp, module_result_path


MODULE_NAME = "offline_shape_alignment"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="offline-shape-alignment",
        description="Diagnose XHand reference keypoints and mesh alignment against MANO.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser(
        "diagnose-xhand-reference",
        help="Generate XHand-to-MANO keypoint and frame diagnostics.",
    )
    diagnose.add_argument("--side", choices=["left", "right", "both"], default="right")
    diagnose.add_argument("--xhand-root", default="assets/hands/xhand")
    diagnose.add_argument("--mano-root", default="assets/hands/mano")
    diagnose.add_argument("--results-root", default="results")
    diagnose.add_argument("--timestamp")
    diagnose.add_argument("--out-json")
    diagnose.add_argument("--out-png")
    diagnose.set_defaults(func=_cmd_diagnose_xhand_reference)

    fit_pose = subparsers.add_parser(
        "fit-xhand-reference-pose",
        help="Fit an interpretable XHand reference qpos before generating MANO diagnostics.",
    )
    fit_pose.add_argument("--side", choices=["left", "right", "both"], default="right")
    fit_pose.add_argument("--xhand-root", default="assets/hands/xhand")
    fit_pose.add_argument("--mano-root", default="assets/hands/mano")
    fit_pose.add_argument("--results-root", default="results")
    fit_pose.add_argument("--timestamp")
    fit_pose.add_argument("--iterations", type=int, default=5)
    fit_pose.add_argument("--initial-step-rad", type=float, default=0.35)
    fit_pose.add_argument("--min-step-rad", type=float, default=0.02)
    fit_pose.add_argument("--out-json")
    fit_pose.add_argument("--out-png")
    fit_pose.add_argument("--out-pose-json")
    fit_pose.set_defaults(func=_cmd_fit_xhand_reference_pose)

    fit_shape = subparsers.add_parser(
        "fit-mano-shape",
        help="Run beta-only MANO shape optimization against aligned XHand reference geometry.",
    )
    fit_shape.add_argument("--side", choices=["left", "right", "both"], default="right")
    fit_shape.add_argument("--xhand-root", default="assets/hands/xhand")
    fit_shape.add_argument("--mano-root", default="assets/hands/mano")
    fit_shape.add_argument("--results-root", default="results")
    fit_shape.add_argument("--timestamp")
    fit_shape.add_argument("--iterations", type=int, default=1000)
    fit_shape.add_argument("--robot-surface-points", type=int, default=2048)
    fit_shape.add_argument("--mano-surface-points", type=int, default=2048)
    fit_shape.add_argument("--learning-rate", type=float, default=0.03)
    fit_shape.add_argument("--beta-l2-weight", type=float, default=1e-4)
    fit_shape.add_argument("--key-decay-steps", type=int, default=2500)
    fit_shape.add_argument("--seed", type=int, default=7)
    fit_shape.add_argument("--reference-pose-iterations", type=int, default=5)
    fit_shape.add_argument("--no-fit-reference-pose", action="store_true")
    fit_shape.add_argument("--out-json")
    fit_shape.add_argument("--out-png")
    fit_shape.add_argument("--out-loss-png")
    fit_shape.add_argument("--out-obj")
    fit_shape.set_defaults(func=_cmd_fit_mano_shape)

    fit_shape_pose = subparsers.add_parser(
        "fit-mano-shape-pose",
        help="Run strongly regularized MANO beta + limited pose residual optimization against XHand.",
    )
    fit_shape_pose.add_argument("--side", choices=["left", "right", "both"], default="right")
    fit_shape_pose.add_argument("--xhand-root", default="assets/hands/xhand")
    fit_shape_pose.add_argument("--mano-root", default="assets/hands/mano")
    fit_shape_pose.add_argument("--results-root", default="results")
    fit_shape_pose.add_argument("--timestamp")
    fit_shape_pose.add_argument("--iterations", type=int, default=600)
    fit_shape_pose.add_argument("--beta-init-iterations", type=int, default=400)
    fit_shape_pose.add_argument("--robot-surface-points", type=int, default=2048)
    fit_shape_pose.add_argument("--mano-surface-points", type=int, default=2048)
    fit_shape_pose.add_argument("--learning-rate", type=float, default=0.03)
    fit_shape_pose.add_argument("--pose-learning-rate", type=float, default=0.01)
    fit_shape_pose.add_argument("--pose-limit-rad", type=float, default=0.25)
    fit_shape_pose.add_argument("--pose-l2-weight", type=float, default=1e-3)
    fit_shape_pose.add_argument("--beta-l2-weight", type=float, default=1e-4)
    fit_shape_pose.add_argument("--key-decay-steps", type=int, default=2500)
    fit_shape_pose.add_argument("--seed", type=int, default=7)
    fit_shape_pose.add_argument("--reference-pose-iterations", type=int, default=5)
    fit_shape_pose.add_argument("--no-fit-reference-pose", action="store_true")
    fit_shape_pose.add_argument("--out-json")
    fit_shape_pose.add_argument("--out-png")
    fit_shape_pose.add_argument("--out-loss-png")
    fit_shape_pose.add_argument("--out-obj")
    fit_shape_pose.set_defaults(func=_cmd_fit_mano_shape_pose)

    args = parser.parse_args(argv)
    args.func(args)


def _cmd_diagnose_xhand_reference(args: argparse.Namespace) -> None:
    sides = ["left", "right"] if args.side == "both" else [args.side]
    timestamp = args.timestamp or make_result_timestamp()
    json_path = module_result_path(
        MODULE_NAME,
        args.out_json or f"xhand_{args.side}_diagnostic.json",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".json",
    )
    png_path = module_result_path(
        MODULE_NAME,
        args.out_png or f"xhand_{args.side}_diagnostic.png",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".png",
    )

    reports = {}
    render_panels = []
    for side in sides:
        urdf_path = default_xhand_urdf_path(args.xhand_root, side)
        robot = load_xhand_reference(urdf_path, side=side)
        mano = load_mano_reference(side, args.mano_root)
        report = diagnose_alignment(robot.keypoints, mano.keypoints, robot.mesh, mano.mesh)
        reports[side] = report
        render_panels.append((report, robot.mesh, mano.mesh))

    payload = {
        "module": MODULE_NAME,
        "command": "diagnose-xhand-reference",
        "sides": reports,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if len(render_panels) == 1:
        report, robot_mesh, mano_mesh = render_panels[0]
        render_alignment_report(report, robot_mesh, mano_mesh, png_path)
    else:
        render_alignment_reports(render_panels, png_path)

    print(f"OK: wrote {json_path}")
    print(f"OK: wrote {png_path}")


def _cmd_fit_xhand_reference_pose(args: argparse.Namespace) -> None:
    sides = ["left", "right"] if args.side == "both" else [args.side]
    timestamp = args.timestamp or make_result_timestamp()
    json_path = module_result_path(
        MODULE_NAME,
        args.out_json or f"xhand_{args.side}_fit_diagnostic.json",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".json",
    )
    png_path = module_result_path(
        MODULE_NAME,
        args.out_png or f"xhand_{args.side}_fit_diagnostic.png",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".png",
    )
    pose_path = module_result_path(
        MODULE_NAME,
        args.out_pose_json or f"xhand_{args.side}_robot_reference_pose.json",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".json",
    )

    reports = {}
    pose_payload = {
        "module": MODULE_NAME,
        "command": "fit-xhand-reference-pose",
        "sides": {},
    }
    render_panels = []
    for side in sides:
        urdf_path = default_xhand_urdf_path(args.xhand_root, side)
        mano = load_mano_reference(side, args.mano_root)
        baseline_robot = load_xhand_reference(urdf_path, side=side)
        baseline_report = diagnose_alignment(baseline_robot.keypoints, mano.keypoints, baseline_robot.mesh, mano.mesh)
        fit = fit_xhand_reference_pose(
            urdf_path,
            side,
            mano.keypoints,
            iterations=args.iterations,
            initial_step_rad=args.initial_step_rad,
            min_step_rad=args.min_step_rad,
        )
        robot = load_xhand_reference(urdf_path, side=side, qpos=fit.qpos)
        report = diagnose_alignment(robot.keypoints, mano.keypoints, robot.mesh, mano.mesh)
        report["reference_pose_fit"] = reference_pose_fit_to_json(fit)
        report["baseline_summary"] = baseline_report["summary"]
        reports[side] = report
        pose_payload["sides"][side] = {
            "hand": "xhand",
            "side": side,
            "urdf_path": str(urdf_path),
            "fit": reference_pose_fit_to_json(fit),
            "baseline_summary": baseline_report["summary"],
            "optimized_summary": report["summary"],
        }
        render_panels.append((report, robot.mesh, mano.mesh))

    payload = {
        "module": MODULE_NAME,
        "command": "fit-xhand-reference-pose",
        "sides": reports,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    with pose_path.open("w", encoding="utf-8") as f:
        json.dump(pose_payload, f, indent=2)

    if len(render_panels) == 1:
        report, robot_mesh, mano_mesh = render_panels[0]
        render_alignment_report(report, robot_mesh, mano_mesh, png_path)
    else:
        render_alignment_reports(render_panels, png_path)

    print(f"OK: wrote {json_path}")
    print(f"OK: wrote {pose_path}")
    print(f"OK: wrote {png_path}")


def _cmd_fit_mano_shape(args: argparse.Namespace) -> None:
    sides = ["left", "right"] if args.side == "both" else [args.side]
    timestamp = args.timestamp or make_result_timestamp()
    json_path = module_result_path(
        MODULE_NAME,
        args.out_json or f"xhand_{args.side}_mano_beta_fit.json",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".json",
    )
    png_path = module_result_path(
        MODULE_NAME,
        args.out_png or f"xhand_{args.side}_mano_beta_fit.png",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".png",
    )
    loss_png_path = module_result_path(
        MODULE_NAME,
        args.out_loss_png or f"xhand_{args.side}_mano_beta_loss.png",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".png",
    )

    config = ShapeOptimizationConfig(
        iterations=args.iterations,
        robot_surface_points=args.robot_surface_points,
        mano_surface_points=args.mano_surface_points,
        learning_rate=args.learning_rate,
        key_decay_steps=args.key_decay_steps,
        beta_l2_weight=args.beta_l2_weight,
        seed=args.seed,
    )

    reports = {}
    render_panels = []
    loss_histories = {}
    obj_paths = {}
    for side in sides:
        result = fit_mano_beta_to_xhand(
            side,
            xhand_root=args.xhand_root,
            mano_root=args.mano_root,
            config=config,
            fit_reference_pose=not args.no_fit_reference_pose,
            reference_pose_iterations=args.reference_pose_iterations,
        )
        reports[side] = shape_optimization_result_to_json(result)
        render_panels.append((result.fixed_frame_report, result.robot_aligned_mesh, result.fitted_mano.mesh))
        loss_histories[side] = list(result.history)

        obj_path = _side_result_path(
            args.out_obj or f"xhand_{args.side}_fitted_mano.obj",
            side,
            side_count=len(sides),
            results_root=args.results_root,
            timestamp=timestamp,
        )
        write_obj(result.fitted_mano.mesh, obj_path)
        obj_paths[side] = str(obj_path)

    payload = {
        "module": MODULE_NAME,
        "command": "fit-mano-shape",
        "sides": reports,
        "outputs": {
            "obj": obj_paths,
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if len(render_panels) == 1:
        report, robot_mesh, mano_mesh = render_panels[0]
        render_alignment_report(report, robot_mesh, mano_mesh, png_path)
    else:
        render_alignment_reports(render_panels, png_path)
    render_loss_histories(loss_histories, loss_png_path)

    print(f"OK: wrote {json_path}")
    for side, obj_path in obj_paths.items():
        print(f"OK: wrote {side} OBJ {obj_path}")
    print(f"OK: wrote {png_path}")
    print(f"OK: wrote {loss_png_path}")


def _cmd_fit_mano_shape_pose(args: argparse.Namespace) -> None:
    sides = ["left", "right"] if args.side == "both" else [args.side]
    timestamp = args.timestamp or make_result_timestamp()
    json_path = module_result_path(
        MODULE_NAME,
        args.out_json or f"xhand_{args.side}_mano_beta_pose_fit.json",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".json",
    )
    png_path = module_result_path(
        MODULE_NAME,
        args.out_png or f"xhand_{args.side}_mano_beta_pose_fit.png",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".png",
    )
    loss_png_path = module_result_path(
        MODULE_NAME,
        args.out_loss_png or f"xhand_{args.side}_mano_beta_pose_loss.png",
        results_root=args.results_root,
        timestamp=timestamp,
        default_suffix=".png",
    )

    config = PoseShapeOptimizationConfig(
        iterations=args.iterations,
        beta_init_iterations=args.beta_init_iterations,
        robot_surface_points=args.robot_surface_points,
        mano_surface_points=args.mano_surface_points,
        learning_rate=args.learning_rate,
        pose_learning_rate=args.pose_learning_rate,
        pose_limit_rad=args.pose_limit_rad,
        pose_l2_weight=args.pose_l2_weight,
        key_decay_steps=args.key_decay_steps,
        beta_l2_weight=args.beta_l2_weight,
        seed=args.seed,
    )

    reports = {}
    render_panels = []
    loss_histories = {}
    obj_paths = {}
    for side in sides:
        result = fit_mano_beta_pose_to_xhand(
            side,
            xhand_root=args.xhand_root,
            mano_root=args.mano_root,
            config=config,
            fit_reference_pose=not args.no_fit_reference_pose,
            reference_pose_iterations=args.reference_pose_iterations,
        )
        reports[side] = pose_shape_optimization_result_to_json(result)
        render_panels.append((result.fixed_frame_report, result.robot_aligned_mesh, result.fitted_mano.mesh))
        loss_histories[side] = list(result.history)

        obj_path = _side_result_path(
            args.out_obj or f"xhand_{args.side}_fitted_mano_beta_pose.obj",
            side,
            side_count=len(sides),
            results_root=args.results_root,
            timestamp=timestamp,
        )
        write_obj(result.fitted_mano.mesh, obj_path)
        obj_paths[side] = str(obj_path)

    payload = {
        "module": MODULE_NAME,
        "command": "fit-mano-shape-pose",
        "sides": reports,
        "outputs": {
            "obj": obj_paths,
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if len(render_panels) == 1:
        report, robot_mesh, mano_mesh = render_panels[0]
        render_alignment_report(report, robot_mesh, mano_mesh, png_path)
    else:
        render_alignment_reports(render_panels, png_path)
    render_loss_histories(loss_histories, loss_png_path)

    print(f"OK: wrote {json_path}")
    for side, obj_path in obj_paths.items():
        print(f"OK: wrote {side} OBJ {obj_path}")
    print(f"OK: wrote {png_path}")
    print(f"OK: wrote {loss_png_path}")


def _side_result_path(
    filename: str,
    side: str,
    *,
    side_count: int,
    results_root: str | Path,
    timestamp: str,
) -> Path:
    path = Path(filename)
    if side_count > 1:
        filename = f"{path.stem}_{side}{path.suffix or '.obj'}"
    return module_result_path(
        MODULE_NAME,
        filename,
        results_root=results_root,
        timestamp=timestamp,
        default_suffix=".obj",
    )


if __name__ == "__main__":
    main()
