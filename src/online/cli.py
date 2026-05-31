from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from online.tactile_mapping import (
    MODULE_NAME,
    load_json_qpos,
    load_online_reference,
    load_tactile_values,
    online_result_to_json,
    project_online_tactile,
    render_online_tactile_map,
    save_online_result_npz,
)
from online.retarget_calibration import (
    PoseIKConfig,
    calibrate_retarget_model,
    default_qpos_ranges,
    infer_xhand_revolute_qpos_names,
    qpos_mapping_to_vector,
    retarget_calibration_to_json,
    sample_qpos_uniform,
    save_retarget_model,
    vector_to_qpos_mapping,
)
from tactile_utils_common.results import make_result_timestamp, module_result_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="online-tactile-map",
        description="Map current XHand qpos and tactile values onto the current posed MANO mesh.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    map_cmd = subparsers.add_parser(
        "map-xhand-tactile",
        help="Project one frame of XHand tactile values onto a posed MANO mesh.",
    )
    map_cmd.add_argument("--side", choices=["left", "right"], required=True)
    map_cmd.add_argument("--alignment-json", required=True)
    map_cmd.add_argument("--projection-npz", required=True)
    map_cmd.add_argument("--qpos-json", required=True)
    map_cmd.add_argument("--tactile-values", required=True)
    map_cmd.add_argument(
        "--mano-frame-transform-json",
        help="Optional JSON 4x4 transform applied to the posed MANO mesh/taxel points in MANO coordinates.",
    )
    map_cmd.add_argument("--xhand-root", default="assets/hands/xhand")
    map_cmd.add_argument("--mano-root", default="assets/hands/mano")
    map_cmd.add_argument("--xhand-urdf")
    map_cmd.add_argument("--retarget-model-npz")
    map_cmd.add_argument("--device", default="cpu")
    map_cmd.add_argument("--dtype", default="float32")
    map_cmd.add_argument("--results-root", default="results")
    map_cmd.add_argument("--timestamp")
    map_cmd.add_argument("--out-json")
    map_cmd.add_argument("--out-npz")
    map_cmd.add_argument("--out-png")
    map_cmd.set_defaults(func=_cmd_map_xhand_tactile)

    calibrate_cmd = subparsers.add_parser(
        "calibrate-retarget",
        help="Sample XHand qpos, solve pose-only MANO IK, and fit a fast qpos-to-MANO retarget model.",
    )
    calibrate_cmd.add_argument("--side", choices=["left", "right"], required=True)
    calibrate_cmd.add_argument("--alignment-json", required=True)
    calibrate_cmd.add_argument("--projection-npz", required=True)
    calibrate_cmd.add_argument("--xhand-root", default="assets/hands/xhand")
    calibrate_cmd.add_argument("--mano-root", default="assets/hands/mano")
    calibrate_cmd.add_argument("--xhand-urdf")
    calibrate_cmd.add_argument("--device", default="cpu")
    calibrate_cmd.add_argument("--dtype", default="float32")
    calibrate_cmd.add_argument("--sample-count", type=int, default=64)
    calibrate_cmd.add_argument("--seed", type=int, default=7)
    calibrate_cmd.add_argument("--qpos-min-json")
    calibrate_cmd.add_argument("--qpos-max-json")
    calibrate_cmd.add_argument("--qpos-samples-json")
    calibrate_cmd.add_argument("--qpos-samples-npz")
    calibrate_cmd.add_argument("--default-qpos-min", type=float, default=0.0)
    calibrate_cmd.add_argument("--default-qpos-max", type=float, default=1.35)
    calibrate_cmd.add_argument("--iterations", type=int, default=160)
    calibrate_cmd.add_argument("--learning-rate", type=float, default=0.03)
    calibrate_cmd.add_argument("--direction-loss-weight", type=float, default=0.15)
    calibrate_cmd.add_argument("--pose-prior-weight", type=float, default=1e-3)
    calibrate_cmd.add_argument("--root-prior-weight", type=float, default=5e-4)
    calibrate_cmd.add_argument("--pose-limit-rad", type=float, default=3.0)
    calibrate_cmd.add_argument("--ridge-lambda", type=float, default=1e-4)
    calibrate_cmd.add_argument("--results-root", default="results")
    calibrate_cmd.add_argument("--timestamp")
    calibrate_cmd.add_argument("--out-model")
    calibrate_cmd.add_argument("--out-json")
    calibrate_cmd.set_defaults(func=_cmd_calibrate_retarget)

    args = parser.parse_args(argv)
    args.func(args)


def _cmd_map_xhand_tactile(args: argparse.Namespace) -> None:
    timestamp = args.timestamp or make_result_timestamp()
    reference = load_online_reference(
        args.alignment_json,
        args.projection_npz,
        args.side,
        xhand_root=args.xhand_root,
        mano_root=args.mano_root,
        xhand_urdf=args.xhand_urdf,
        retarget_model_npz=args.retarget_model_npz,
        device=args.device,
        dtype=args.dtype,
    )
    qpos = load_json_qpos(args.qpos_json)
    tactile_values = load_tactile_values(args.tactile_values)
    mano_frame_transform = _load_transform_json(args.mano_frame_transform_json) if args.mano_frame_transform_json else None
    result = project_online_tactile(reference, qpos, tactile_values, mano_frame_transform=mano_frame_transform)

    npz_path = _result_path(
        args.out_npz or f"xhand_{args.side}_online_tactile_map.npz",
        args.results_root,
        timestamp,
        ".npz",
    )
    json_path = _result_path(
        args.out_json or f"xhand_{args.side}_online_tactile_map.json",
        args.results_root,
        timestamp,
        ".json",
    )
    png_path = _result_path(
        args.out_png or f"xhand_{args.side}_online_tactile_map.png",
        args.results_root,
        timestamp,
        ".png",
    )

    save_online_result_npz(result, npz_path)
    render_online_tactile_map(result, png_path)
    payload = online_result_to_json(
        result,
        outputs={
            "npz": str(npz_path),
            "png": str(png_path),
        },
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"OK: wrote {npz_path}")
    print(f"OK: wrote {png_path}")
    print(f"OK: wrote {json_path}")


def _cmd_calibrate_retarget(args: argparse.Namespace) -> None:
    timestamp = args.timestamp or make_result_timestamp()
    reference = load_online_reference(
        args.alignment_json,
        args.projection_npz,
        args.side,
        xhand_root=args.xhand_root,
        mano_root=args.mano_root,
        xhand_urdf=args.xhand_urdf,
        device=args.device,
        dtype=args.dtype,
    )
    qpos_names = infer_xhand_revolute_qpos_names(reference.xhand_urdf_path, args.side)
    qpos_samples = _load_or_sample_qpos(args, reference, qpos_names)
    ik_config = PoseIKConfig(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        direction_loss_weight=args.direction_loss_weight,
        pose_prior_weight=args.pose_prior_weight,
        root_prior_weight=args.root_prior_weight,
        pose_limit_rad=args.pose_limit_rad,
    )
    calibration = calibrate_retarget_model(
        reference,
        qpos_samples,
        qpos_names=qpos_names,
        ik_config=ik_config,
        ridge_lambda=args.ridge_lambda,
    )

    model_path = _result_path(
        args.out_model or f"xhand_{args.side}_mano_retarget_model.npz",
        args.results_root,
        timestamp,
        ".npz",
    )
    json_path = _result_path(
        args.out_json or f"xhand_{args.side}_mano_retarget_model.json",
        args.results_root,
        timestamp,
        ".json",
    )
    save_retarget_model(calibration.model, model_path, calibration=calibration)
    payload = retarget_calibration_to_json(
        calibration,
        outputs={
            "model_npz": str(model_path),
            "json": str(json_path),
        },
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"OK: wrote {model_path}")
    print(f"OK: wrote {json_path}")


def _result_path(filename: str, results_root: str | Path, timestamp: str, default_suffix: str) -> Path:
    return module_result_path(
        MODULE_NAME,
        filename,
        results_root=results_root,
        timestamp=timestamp,
        default_suffix=default_suffix,
    )


def _load_transform_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "mano_frame_transform" in payload:
        payload = payload["mano_frame_transform"]
    return payload


def _load_or_sample_qpos(args: argparse.Namespace, reference, qpos_names: tuple[str, ...]) -> list[dict[str, float]]:
    if args.qpos_samples_json:
        return _load_qpos_samples_json(args.qpos_samples_json)
    if args.qpos_samples_npz:
        return _load_qpos_samples_npz(args.qpos_samples_npz, qpos_names, reference.reference_qpos)
    qpos_min, qpos_max = default_qpos_ranges(
        qpos_names,
        qpos_min=args.default_qpos_min,
        qpos_max=args.default_qpos_max,
        reference_qpos=reference.reference_qpos,
    )
    if args.qpos_min_json:
        qpos_min.update(_load_mapping_json(args.qpos_min_json, preferred_key="qpos_min"))
    if args.qpos_max_json:
        qpos_max.update(_load_mapping_json(args.qpos_max_json, preferred_key="qpos_max"))
    return sample_qpos_uniform(qpos_names, qpos_min, qpos_max, count=args.sample_count, seed=args.seed)


def _load_qpos_samples_json(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("qpos_samples", payload.get("samples"))
    if not isinstance(payload, list):
        raise ValueError("qpos samples JSON must be a list or contain qpos_samples")
    out = []
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("qpos"), dict):
            item = item["qpos"]
        if not isinstance(item, dict):
            raise ValueError("each qpos sample must be a mapping or contain a qpos mapping")
        out.append({str(name): float(value) for name, value in item.items()})
    return out


def _load_qpos_samples_npz(path: str | Path, default_names: tuple[str, ...], reference_qpos: dict[str, float]) -> list[dict[str, float]]:
    with np.load(path, allow_pickle=False) as payload:
        if "qpos_values" not in payload.files:
            raise ValueError("qpos samples npz must contain qpos_values")
        values = payload["qpos_values"]
        names = (
            tuple(str(item) for item in np.asarray(payload["qpos_names"]).tolist())
            if "qpos_names" in payload.files
            else default_names
        )
    qpos_values = np.asarray(values, dtype=np.float64)
    if qpos_values.ndim != 2:
        raise ValueError(f"qpos_values must have shape (N, J), got {qpos_values.shape}")
    if qpos_values.shape[1] != len(names):
        raise ValueError("qpos_values width does not match qpos_names")
    samples = [vector_to_qpos_mapping(row, names) for row in qpos_values]
    if names != default_names:
        samples = [
            {
                name: float(qpos_mapping_to_vector(sample, default_names, reference_qpos=reference_qpos)[idx])
                for idx, name in enumerate(default_names)
            }
            for sample in samples
        ]
    return samples


def _load_mapping_json(path: str | Path, *, preferred_key: str) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get(preferred_key), dict):
        payload = payload[preferred_key]
    if not isinstance(payload, dict):
        raise ValueError(f"{preferred_key} JSON must contain a mapping")
    return {str(name): float(value) for name, value in payload.items()}


if __name__ == "__main__":
    main()
