#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from online.tactile_mapping import (  # noqa: E402
    load_online_reference,
    online_result_to_json,
    project_online_tactile,
    render_online_tactile_map,
    save_online_result_npz,
)
from offline.tactile_to_mano_projection import build_xhand_tactile_layout  # noqa: E402
from sub_modules.offline_shape_alignment.alignment import apply_similarity_to_points  # noqa: E402
from sub_modules.offline_shape_alignment.xhand import default_xhand_urdf_path, load_xhand_reference  # noqa: E402
from tactile_utils_common.results import make_result_timestamp, module_result_path  # noqa: E402


MODULE_NAME = "online_real_data_test"
DEFAULT_DATA_DIR = Path("/home/jingwei/bjw/data/realman_xhand_data_test_with_tactile/process_data/pick_place_0")
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_XHAND_ROOT = REPO_ROOT / "assets" / "hands" / "xhand"
DEFAULT_MANO_ROOT = REPO_ROOT / "assets" / "hands" / "mano"

XHAND_12_DOF_SUFFIXES = (
    "thumb_bend_joint",
    "thumb_rota_joint1",
    "thumb_rota_joint2",
    "index_bend_joint",
    "index_joint1",
    "index_joint2",
    "mid_joint1",
    "mid_joint2",
    "ring_joint1",
    "ring_joint2",
    "pinky_joint1",
    "pinky_joint2",
)


@dataclass(frozen=True)
class RealFrameSample:
    array_index: int
    step: int
    time_s: float
    wall_time: float
    color_path: str | None
    depth_path: str | None
    qpos_vector: np.ndarray
    qpos: dict[str, float]
    tactile_raw: np.ndarray
    tactile_values: np.ndarray
    tactile_sum: float
    tactile_max: float
    wrist_transform_base: np.ndarray | None


@dataclass(frozen=True)
class XHandTactilePose:
    side: str
    vertices: np.ndarray
    faces: np.ndarray
    taxel_points: np.ndarray
    tactile_values: np.ndarray
    qpos: dict[str, float]
    sensor_names: tuple[str, ...]
    semantic_regions: tuple[str, ...]
    taxel_indices: np.ndarray
    alignment_transform: np.ndarray
    coordinate_frame: str
    urdf_path: Path
    tactile_dir: Path


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    results_root = Path(args.results_root)
    timestamp = args.timestamp or make_result_timestamp()
    alignment_json = Path(args.alignment_json) if args.alignment_json else _default_alignment_json(REPO_ROOT)
    projection_npz = (
        Path(args.projection_npz)
        if args.projection_npz
        else _default_projection_npz(REPO_ROOT, args.side, prefer_no_runtime=not args.allow_runtime_probe_projection)
    )

    frames = _load_real_frames(data_dir)
    raw_baseline = _raw_tactile_baseline(frames, args.side, args.baseline_frames) if args.tactile_mode == "delta-norm" else None
    samples = _build_samples(
        frames,
        side=args.side,
        qpos_source=args.qpos_source,
        tactile_mode=args.tactile_mode,
        raw_baseline=raw_baseline,
        arm_rotation_mode=args.arm_rotation_mode,
    )
    if len(samples) < args.keyframes:
        raise ValueError(f"need at least {args.keyframes} valid frames, found {len(samples)}")

    selected_indices, selected_distances = _select_keyframes_by_qpos(
        np.stack([sample.qpos_vector for sample in samples], axis=0),
        args.keyframes,
    )
    selected_pairs = [(idx, samples[idx]) for idx in sorted(selected_indices, key=lambda i: samples[i].step)]
    selected = [sample for _, sample in selected_pairs]

    print(f"data_dir: {data_dir}")
    print(f"valid frames: {len(samples)}")
    print(f"alignment_json: {alignment_json}")
    print(f"projection_npz: {projection_npz}")
    print("selected steps:", ", ".join(str(sample.step) for sample in selected))
    wrist_reference = _select_wrist_reference(samples, args.wrist_pose_source)
    if wrist_reference is not None:
        print(f"wrist pose source: {args.wrist_pose_source} reference_step={wrist_reference.step}")
    else:
        print("wrist pose source: none")
    if args.dry_run:
        return

    reference = load_online_reference(
        alignment_json,
        projection_npz,
        args.side,
        xhand_root=args.xhand_root,
        mano_root=args.mano_root,
        retarget_model_npz=args.retarget_model_npz,
        device=args.device,
        dtype=args.dtype,
    )
    if reference.projection.taxel_count != selected[0].tactile_values.shape[0]:
        raise ValueError(
            f"projection taxel count {reference.projection.taxel_count} does not match real tactile count "
            f"{selected[0].tactile_values.shape[0]}"
        )

    mano_png_paths: list[Path] = []
    xhand_png_paths: list[Path] = []
    comparison_png_paths: list[Path] = []
    overlay_png_paths: list[Path] = []
    frame_summaries: list[dict[str, Any]] = []
    for rank, (sample_idx, sample) in enumerate(selected_pairs):
        xhand_root_delta = _xhand_root_delta(wrist_reference, sample)
        mano_frame_transform = _mano_frame_transform_from_xhand_delta(reference.robot_to_mano, xhand_root_delta)
        xhand_alignment_transform = reference.robot_to_mano @ xhand_root_delta
        result = project_online_tactile(
            reference,
            sample.qpos,
            sample.tactile_values,
            mano_frame_transform=mano_frame_transform,
        )
        xhand_pose = _build_xhand_tactile_pose(
            side=args.side,
            xhand_root=args.xhand_root,
            qpos=sample.qpos,
            tactile_values=sample.tactile_values,
            alignment_transform=xhand_alignment_transform,
            coordinate_frame="mano_frame_with_relative_wrist",
        )
        stem = f"{args.output_prefix}_{args.side}_frame_{sample.step:06d}"
        npz_path = module_result_path(MODULE_NAME, f"{stem}.npz", results_root=results_root, timestamp=timestamp)
        png_path = module_result_path(MODULE_NAME, f"{stem}.png", results_root=results_root, timestamp=timestamp)
        json_path = module_result_path(MODULE_NAME, f"{stem}.json", results_root=results_root, timestamp=timestamp)
        xhand_npz_path = module_result_path(
            MODULE_NAME,
            f"{stem}_xhand.npz",
            results_root=results_root,
            timestamp=timestamp,
        )
        xhand_png_path = module_result_path(
            MODULE_NAME,
            f"{stem}_xhand.png",
            results_root=results_root,
            timestamp=timestamp,
        )
        comparison_png_path = module_result_path(
            MODULE_NAME,
            f"{stem}_mano_xhand.png",
            results_root=results_root,
            timestamp=timestamp,
        )
        overlay_png_path = module_result_path(
            MODULE_NAME,
            f"{stem}_mano_xhand_overlay.png",
            results_root=results_root,
            timestamp=timestamp,
        )

        save_online_result_npz(result, npz_path)
        render_online_tactile_map(
            result,
            png_path,
            title=f"MANO online | {args.side} step={sample.step} keyframe {rank + 1}/{len(selected)}",
        )
        _save_xhand_tactile_npz(xhand_pose, xhand_npz_path)
        _render_xhand_tactile_map(
            xhand_pose,
            xhand_png_path,
            title=f"XHand qpos | {args.side} step={sample.step} keyframe {rank + 1}/{len(selected)}",
        )
        _write_side_by_side(
            (png_path, xhand_png_path),
            ("online MANO tactile map", "posed XHand tactile map"),
            comparison_png_path,
        )
        _render_mano_xhand_overlay(
            result,
            xhand_pose,
            overlay_png_path,
            title=f"MANO/XHand overlay | {args.side} step={sample.step}",
        )
        frame_outputs = {
            "mano_npz": str(npz_path),
            "mano_png": str(png_path),
            "xhand_npz": str(xhand_npz_path),
            "xhand_png": str(xhand_png_path),
            "comparison_png": str(comparison_png_path),
            "overlay_png": str(overlay_png_path),
        }
        payload = online_result_to_json(result, outputs=frame_outputs)
        payload["real_data"] = _sample_payload(sample, rank, selected_distances.get(sample_idx))
        payload["xhand"] = {
            "vertex_count": int(xhand_pose.vertices.shape[0]),
            "face_count": int(xhand_pose.faces.shape[0]),
            "taxel_count": int(xhand_pose.taxel_points.shape[0]),
            "coordinate_frame": xhand_pose.coordinate_frame,
            "alignment_transform": xhand_pose.alignment_transform.tolist(),
            "urdf_path": str(xhand_pose.urdf_path),
            "tactile_dir": str(xhand_pose.tactile_dir),
        }
        payload["wrist_pose"] = _wrist_payload(wrist_reference, sample, xhand_root_delta, mano_frame_transform)
        _write_json(json_path, payload)

        frame_summary = dict(payload["real_data"])
        frame_summary["outputs"] = frame_outputs
        frame_summary["wrist_pose"] = payload["wrist_pose"]
        frame_summaries.append(frame_summary)
        mano_png_paths.append(png_path)
        xhand_png_paths.append(xhand_png_path)
        comparison_png_paths.append(comparison_png_path)
        overlay_png_paths.append(overlay_png_path)
        print(f"wrote {png_path}")
        print(f"wrote {xhand_png_path}")
        print(f"wrote {comparison_png_path}")
        print(f"wrote {overlay_png_path}")

    mano_montage_path = None
    xhand_montage_path = None
    comparison_montage_path = None
    overlay_montage_path = None
    if not args.no_montage:
        mano_montage_path = module_result_path(
            MODULE_NAME,
            f"{args.output_prefix}_{args.side}_keyframes_montage.png",
            results_root=results_root,
            timestamp=timestamp,
        )
        xhand_montage_path = module_result_path(
            MODULE_NAME,
            f"{args.output_prefix}_{args.side}_xhand_keyframes_montage.png",
            results_root=results_root,
            timestamp=timestamp,
        )
        comparison_montage_path = module_result_path(
            MODULE_NAME,
            f"{args.output_prefix}_{args.side}_mano_xhand_keyframes_montage.png",
            results_root=results_root,
            timestamp=timestamp,
        )
        overlay_montage_path = module_result_path(
            MODULE_NAME,
            f"{args.output_prefix}_{args.side}_mano_xhand_overlay_keyframes_montage.png",
            results_root=results_root,
            timestamp=timestamp,
        )
        _write_montage(
            mano_png_paths,
            [f"step {sample.step}" for sample in selected],
            mano_montage_path,
        )
        _write_montage(
            xhand_png_paths,
            [f"step {sample.step}" for sample in selected],
            xhand_montage_path,
        )
        _write_montage(
            comparison_png_paths,
            [f"step {sample.step}" for sample in selected],
            comparison_montage_path,
        )
        _write_montage(
            overlay_png_paths,
            [f"step {sample.step}" for sample in selected],
            overlay_montage_path,
        )
        print(f"wrote {mano_montage_path}")
        print(f"wrote {xhand_montage_path}")
        print(f"wrote {comparison_montage_path}")
        print(f"wrote {overlay_montage_path}")

    summary_path = module_result_path(
        MODULE_NAME,
        f"{args.output_prefix}_{args.side}_summary.json",
        results_root=results_root,
        timestamp=timestamp,
    )
    summary = {
        "module": MODULE_NAME,
        "side": args.side,
        "data_dir": str(data_dir),
        "alignment_json": str(alignment_json),
        "projection_npz": str(projection_npz),
        "retarget_model_npz": None if args.retarget_model_npz is None else str(Path(args.retarget_model_npz)),
        "xhand_root": str(Path(args.xhand_root)),
        "mano_root": str(Path(args.mano_root)),
        "qpos_source": args.qpos_source,
        "wrist_pose_source": args.wrist_pose_source,
        "arm_rotation_mode": args.arm_rotation_mode,
        "wrist_reference_step": None if wrist_reference is None else int(wrist_reference.step),
        "qpos_joint_order": [f"{args.side}_hand_{suffix}" for suffix in XHAND_12_DOF_SUFFIXES],
        "tactile_mode": args.tactile_mode,
        "baseline_frames": int(args.baseline_frames) if args.tactile_mode == "delta-norm" else 0,
        "valid_frame_count": len(samples),
        "selected_frame_count": len(selected),
        "selected_steps": [sample.step for sample in selected],
        "projection_taxel_count": int(reference.projection.taxel_count),
        "projection_sensor_order": _projection_sensor_order(reference.projection.sensor_names),
        "outputs": {
            "summary_json": str(summary_path),
            "montage_png": None if mano_montage_path is None else str(mano_montage_path),
            "mano_montage_png": None if mano_montage_path is None else str(mano_montage_path),
            "xhand_montage_png": None if xhand_montage_path is None else str(xhand_montage_path),
            "comparison_montage_png": None if comparison_montage_path is None else str(comparison_montage_path),
            "overlay_montage_png": None if overlay_montage_path is None else str(overlay_montage_path),
            "frame_png": [str(path) for path in mano_png_paths],
            "mano_frame_png": [str(path) for path in mano_png_paths],
            "xhand_frame_png": [str(path) for path in xhand_png_paths],
            "comparison_frame_png": [str(path) for path in comparison_png_paths],
            "overlay_frame_png": [str(path) for path in overlay_png_paths],
        },
        "frames": frame_summaries,
    }
    _write_json(summary_path, summary)
    print(f"wrote {summary_path}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run online XHand tactile-to-MANO mapping on real pick_place_0 frames and visualize 5 qpos-spread keyframes."
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--alignment-json")
    parser.add_argument("--projection-npz")
    parser.add_argument("--retarget-model-npz")
    parser.add_argument("--xhand-root", default=str(DEFAULT_XHAND_ROOT))
    parser.add_argument("--mano-root", default=str(DEFAULT_MANO_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--timestamp")
    parser.add_argument("--output-prefix", default="pick_place_0_online_real")
    parser.add_argument("--keyframes", type=int, default=5)
    parser.add_argument("--qpos-source", choices=["actual_joint_rad", "command_joint_rad"], default="actual_joint_rad")
    parser.add_argument(
        "--wrist-pose-source",
        choices=["arm-ee-relative", "none"],
        default="arm-ee-relative",
        help="Use the arm end-effector pose as a relative XHand wrist/root transform before rendering MANO/XHand.",
    )
    parser.add_argument(
        "--arm-rotation-mode",
        choices=["rpy", "rotvec"],
        default="rpy",
        help="Interpret left/right arm ee_rot_base as roll-pitch-yaw or as an axis-angle rotation vector.",
    )
    parser.add_argument(
        "--tactile-mode",
        choices=["raw-norm", "raw-z", "raw-z-positive", "delta-norm"],
        default="raw-norm",
        help="Convert each 3D raw_force taxel vector to one scalar tactile value.",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=20,
        help="Number of leading frames used as raw-force median baseline for --tactile-mode delta-norm.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--allow-runtime-probe-projection", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only load data and print selected keyframes.")
    parser.add_argument("--no-montage", action="store_true")
    return parser.parse_args(argv)


def _load_real_frames(data_dir: Path) -> list[Mapping[str, Any]]:
    npy_path = data_dir / "data.npy"
    if npy_path.exists():
        payload = np.load(npy_path, allow_pickle=True)
        return [dict(item) for item in payload.tolist()]

    json_path = data_dir / "data.json"
    if json_path.exists():
        frames = []
        with json_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        return frames

    raise FileNotFoundError(f"expected {npy_path} or {json_path}")


def _build_samples(
    frames: Sequence[Mapping[str, Any]],
    *,
    side: str,
    qpos_source: str,
    tactile_mode: str,
    raw_baseline: np.ndarray | None,
    arm_rotation_mode: str,
) -> list[RealFrameSample]:
    samples: list[RealFrameSample] = []
    for array_index, frame in enumerate(frames):
        side_payload = frame.get(side)
        if not isinstance(side_payload, Mapping):
            continue
        hand = side_payload.get("hand")
        if not isinstance(hand, Mapping):
            continue
        arm = side_payload.get("arm")
        qpos_vector = np.asarray(hand.get(qpos_source), dtype=np.float64)
        if qpos_vector.shape != (len(XHAND_12_DOF_SUFFIXES),):
            continue
        tactile_raw = _extract_tactile_raw(hand)
        tactile_values = _tactile_values_from_raw(tactile_raw, tactile_mode, raw_baseline)
        realsense = frame.get("realsense", {})
        samples.append(
            RealFrameSample(
                array_index=array_index,
                step=int(frame.get("step", array_index)),
                time_s=float(frame.get("time", 0.0)),
                wall_time=float(frame.get("wall_time", 0.0)),
                color_path=_optional_str(realsense.get("color_path") if isinstance(realsense, Mapping) else None),
                depth_path=_optional_str(realsense.get("depth_path") if isinstance(realsense, Mapping) else None),
                qpos_vector=qpos_vector,
                qpos=_qpos_mapping(qpos_vector, side),
                tactile_raw=tactile_raw,
                tactile_values=tactile_values,
                tactile_sum=float(tactile_values.sum()),
                tactile_max=float(tactile_values.max()) if tactile_values.size else 0.0,
                wrist_transform_base=_arm_pose_transform(arm, arm_rotation_mode) if isinstance(arm, Mapping) else None,
            )
        )
    return samples


def _extract_tactile_raw(hand: Mapping[str, Any]) -> np.ndarray:
    sensors = hand.get("tactile")
    if not isinstance(sensors, Sequence) or len(sensors) != 5:
        raise ValueError("expected 5 tactile sensors in real data frame")
    arrays = []
    for sensor in sensors:
        if not isinstance(sensor, Mapping):
            raise ValueError("tactile sensor entry must be a mapping")
        raw = np.asarray(sensor.get("raw_force"), dtype=np.float64)
        if raw.shape != (120, 3):
            raise ValueError(f"expected tactile raw_force shape (120, 3), got {raw.shape}")
        arrays.append(raw)
    return np.concatenate(arrays, axis=0)


def _raw_tactile_baseline(frames: Sequence[Mapping[str, Any]], side: str, baseline_frames: int) -> np.ndarray:
    if baseline_frames <= 0:
        return np.zeros((600, 3), dtype=np.float64)
    raw_values = []
    for frame in frames:
        side_payload = frame.get(side)
        hand = side_payload.get("hand") if isinstance(side_payload, Mapping) else None
        if isinstance(hand, Mapping):
            raw_values.append(_extract_tactile_raw(hand))
        if len(raw_values) >= baseline_frames:
            break
    if not raw_values:
        raise ValueError("could not compute tactile baseline because no valid tactile frames were found")
    return np.median(np.stack(raw_values, axis=0), axis=0)


def _tactile_values_from_raw(raw: np.ndarray, mode: str, baseline: np.ndarray | None) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if mode == "delta-norm":
        if baseline is not None:
            values = values - baseline
        return np.linalg.norm(values, axis=1)
    if mode == "raw-norm":
        return np.linalg.norm(values, axis=1)
    if mode == "raw-z":
        return values[:, 2].copy()
    if mode == "raw-z-positive":
        return np.maximum(values[:, 2], 0.0)
    raise ValueError(f"unsupported tactile mode {mode!r}")


def _arm_pose_transform(arm: Mapping[str, Any], rotation_mode: str) -> np.ndarray | None:
    position = np.asarray(arm.get("ee_pos_base"), dtype=np.float64)
    rotation_values = np.asarray(arm.get("ee_rot_base"), dtype=np.float64)
    if position.shape != (3,) or rotation_values.shape != (3,):
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_vector(rotation_values, rotation_mode)
    transform[:3, 3] = position
    return transform


def _rotation_from_vector(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if mode == "rotvec":
        return _axis_angle_matrix(values)
    if mode == "rpy":
        roll, pitch, yaw = values
        return _rot_z(float(yaw)) @ _rot_y(float(pitch)) @ _rot_x(float(roll))
    raise ValueError(f"unsupported arm rotation mode {mode!r}")


def _qpos_mapping(values: np.ndarray, side: str) -> dict[str, float]:
    return {f"{side}_hand_{suffix}": float(value) for suffix, value in zip(XHAND_12_DOF_SUFFIXES, values)}


def _select_keyframes_by_qpos(qpos: np.ndarray, count: int) -> tuple[list[int], dict[int, float]]:
    if count <= 0:
        raise ValueError("keyframes must be positive")
    if qpos.shape[0] <= count:
        return list(range(qpos.shape[0])), {idx: 0.0 for idx in range(qpos.shape[0])}

    scale = np.ptp(qpos, axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (qpos - np.median(qpos, axis=0)) / scale
    center_dist = np.linalg.norm(normalized, axis=1)
    selected = [int(np.argmax(center_dist))]
    min_dist = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
    min_dist[selected[0]] = -1.0

    while len(selected) < count:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        dist_to_next = np.linalg.norm(normalized - normalized[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist_to_next)
        min_dist[selected] = -1.0

    distances = {}
    for idx in selected:
        others = [other for other in selected if other != idx]
        if not others:
            distances[idx] = 0.0
        else:
            distances[idx] = float(np.min(np.linalg.norm(normalized[idx] - normalized[others], axis=1)))
    return selected, distances


def _default_alignment_json(repo_root: Path) -> Path:
    root = repo_root / "results" / "offline_shape_alignment"
    patterns = (
        "*_xhand_both_mano_beta_pose_fit_limit015.json",
        "*_xhand_both_mano_beta_pose_fit.json",
        "*_xhand_both_mano_beta_fit.json",
    )
    for pattern in patterns:
        candidates = sorted(root.glob(pattern))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f"could not find an offline alignment JSON under {root}")


def _default_projection_npz(repo_root: Path, side: str, *, prefer_no_runtime: bool) -> Path:
    root = repo_root / "results" / "tactile_to_mano_projection"
    patterns = (
        f"**/*_xhand_both_graph_preserving_no_block_sensor_to_mano_weights_{side}.npz",
        f"**/*_xhand_both_graph_preserving_sensor_to_mano_weights_{side}.npz",
        f"**/*_sensor_to_mano_weights_{side}.npz",
    )
    fallback: list[Path] = []
    for pattern in patterns:
        candidates = sorted(root.glob(pattern))
        if not candidates:
            continue
        fallback.extend(candidates)
        non_runtime = [path for path in candidates if "runtime_probe" not in path.name]
        usable = non_runtime if prefer_no_runtime and non_runtime else candidates
        if usable:
            return max(usable, key=lambda path: path.stat().st_mtime)
    if fallback:
        return max(fallback, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f"could not find a {side} offline projection weights npz under {root}")


def _projection_sensor_order(sensor_names: Sequence[str]) -> list[dict[str, Any]]:
    order: list[dict[str, Any]] = []
    current_name = None
    start = 0
    for idx, name in enumerate(sensor_names):
        if current_name is None:
            current_name = name
            start = idx
        elif name != current_name:
            order.append({"name": current_name, "start": start, "count": idx - start})
            current_name = name
            start = idx
    if current_name is not None:
        order.append({"name": current_name, "start": start, "count": len(sensor_names) - start})
    return order


def _sample_payload(sample: RealFrameSample, rank: int, qpos_distance: float | None) -> dict[str, Any]:
    payload = {
        "rank": int(rank),
        "array_index": int(sample.array_index),
        "step": int(sample.step),
        "time_s": float(sample.time_s),
        "wall_time": float(sample.wall_time),
        "color_path": sample.color_path,
        "depth_path": sample.depth_path,
        "qpos_min_distance_to_selected": None if qpos_distance is None else float(qpos_distance),
        "qpos_values": sample.qpos_vector.astype(float).tolist(),
        "qpos": sample.qpos,
        "tactile_sum": float(sample.tactile_sum),
        "tactile_max": float(sample.tactile_max),
        "tactile_nonzero_count": int(np.count_nonzero(sample.tactile_values)),
    }
    if sample.wrist_transform_base is not None:
        payload["wrist_transform_base"] = sample.wrist_transform_base.tolist()
    return payload


def _select_wrist_reference(samples: Sequence[RealFrameSample], source: str) -> RealFrameSample | None:
    if source == "none":
        return None
    if source != "arm-ee-relative":
        raise ValueError(f"unsupported wrist pose source {source!r}")
    for sample in samples:
        if sample.wrist_transform_base is not None:
            return sample
    return None


def _xhand_root_delta(reference_sample: RealFrameSample | None, sample: RealFrameSample) -> np.ndarray:
    if reference_sample is None or reference_sample.wrist_transform_base is None or sample.wrist_transform_base is None:
        return np.eye(4, dtype=np.float64)
    return np.linalg.inv(reference_sample.wrist_transform_base) @ sample.wrist_transform_base


def _mano_frame_transform_from_xhand_delta(robot_to_mano: np.ndarray, xhand_root_delta: np.ndarray) -> np.ndarray:
    robot_to_mano = np.asarray(robot_to_mano, dtype=np.float64)
    xhand_root_delta = np.asarray(xhand_root_delta, dtype=np.float64)
    return robot_to_mano @ xhand_root_delta @ np.linalg.inv(robot_to_mano)


def _wrist_payload(
    reference_sample: RealFrameSample | None,
    sample: RealFrameSample,
    xhand_root_delta: np.ndarray,
    mano_frame_transform: np.ndarray,
) -> dict[str, Any]:
    rotation = np.asarray(xhand_root_delta[:3, :3], dtype=np.float64)
    angle = _rotation_angle(rotation)
    translation = np.asarray(xhand_root_delta[:3, 3], dtype=np.float64)
    return {
        "source": "none" if reference_sample is None else "arm-ee-relative",
        "reference_step": None if reference_sample is None else int(reference_sample.step),
        "current_step": int(sample.step),
        "xhand_root_delta": np.asarray(xhand_root_delta, dtype=np.float64).tolist(),
        "mano_frame_transform": np.asarray(mano_frame_transform, dtype=np.float64).tolist(),
        "relative_translation_norm_m": float(np.linalg.norm(translation)),
        "relative_rotation_deg": float(angle * 180.0 / np.pi),
    }


def _build_xhand_tactile_pose(
    *,
    side: str,
    xhand_root: str | Path,
    qpos: Mapping[str, float],
    tactile_values: np.ndarray,
    alignment_transform: np.ndarray,
    coordinate_frame: str,
) -> XHandTactilePose:
    root = Path(xhand_root)
    urdf_path = default_xhand_urdf_path(root, side)
    tactile_dir = root / "tactile sensor" / "xhand_tac"
    xhand = load_xhand_reference(urdf_path, side, qpos=qpos)
    layout = build_xhand_tactile_layout(urdf_path, tactile_dir, side, qpos=qpos)
    tactile = np.asarray(tactile_values, dtype=np.float64)
    if layout.points_robot.shape[0] != tactile.shape[0]:
        raise ValueError(
            f"XHand taxel count {layout.points_robot.shape[0]} does not match tactile value count {tactile.shape[0]}"
        )
    alignment_transform = np.asarray(alignment_transform, dtype=np.float64)
    if alignment_transform.shape != (4, 4):
        raise ValueError(f"alignment_transform must have shape (4, 4), got {alignment_transform.shape}")
    return XHandTactilePose(
        side=side,
        vertices=apply_similarity_to_points(alignment_transform, np.asarray(xhand.mesh.vertices, dtype=np.float64)),
        faces=np.asarray(xhand.mesh.faces, dtype=np.int64),
        taxel_points=apply_similarity_to_points(alignment_transform, np.asarray(layout.points_robot, dtype=np.float64)),
        tactile_values=tactile,
        qpos={str(name): float(value) for name, value in qpos.items()},
        sensor_names=tuple(layout.sensor_names),
        semantic_regions=tuple(layout.semantic_regions),
        taxel_indices=np.asarray(layout.taxel_indices, dtype=np.int64),
        alignment_transform=alignment_transform,
        coordinate_frame=coordinate_frame,
        urdf_path=urdf_path,
        tactile_dir=tactile_dir,
    )


def _save_xhand_tactile_npz(xhand_pose: XHandTactilePose, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qpos_names = np.asarray(sorted(xhand_pose.qpos))
    qpos_values = np.asarray([xhand_pose.qpos[name] for name in qpos_names], dtype=np.float64)
    np.savez_compressed(
        out_path,
        xhand_vertices=xhand_pose.vertices.astype(np.float64),
        xhand_faces=xhand_pose.faces.astype(np.int64),
        xhand_taxel_points=xhand_pose.taxel_points.astype(np.float64),
        tactile_values=xhand_pose.tactile_values.astype(np.float64),
        qpos_names=qpos_names,
        qpos_values=qpos_values,
        sensor_names=np.asarray(xhand_pose.sensor_names),
        semantic_regions=np.asarray(xhand_pose.semantic_regions),
        taxel_indices=xhand_pose.taxel_indices.astype(np.int64),
        alignment_transform=xhand_pose.alignment_transform.astype(np.float64),
        coordinate_frame=np.asarray(xhand_pose.coordinate_frame),
    )


def _render_xhand_tactile_map(
    xhand_pose: XHandTactilePose,
    out_path: str | Path,
    *,
    width: int = 900,
    height: int = 700,
    title: str | None = None,
) -> None:
    from PIL import Image, ImageDraw

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    margin = 44
    bounds_parts = [xhand_pose.taxel_points]
    if xhand_pose.vertices.size:
        bounds_parts.append(xhand_pose.vertices[::_mesh_stride(xhand_pose.vertices.shape[0], 60000)])
    bounds_points = np.concatenate(bounds_parts, axis=0)
    projected_bounds, _ = _project_xhand_view(bounds_points, xhand_pose.side)
    xy_min = projected_bounds.min(axis=0)
    xy_max = projected_bounds.max(axis=0)
    extent = np.maximum(xy_max - xy_min, 1e-9)
    scale = min((width - 2 * margin) / extent[0], (height - 2 * margin) / extent[1])
    center = (xy_min + xy_max) / 2.0

    def to_screen(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy, z = _project_xhand_view(points, xhand_pose.side)
        screen = np.empty_like(xy)
        screen[:, 0] = (xy[:, 0] - center[0]) * scale + width / 2.0
        screen[:, 1] = height / 2.0 - (xy[:, 1] - center[1]) * scale
        return screen, z

    if xhand_pose.vertices.size and xhand_pose.faces.size:
        vertices_xy, vertex_depth = to_screen(xhand_pose.vertices)
        face_step = _mesh_stride(xhand_pose.faces.shape[0], 26000)
        faces = xhand_pose.faces[::face_step]
        face_depth = vertex_depth[faces].mean(axis=1)
        for face_idx in np.argsort(face_depth):
            face = faces[int(face_idx)]
            polygon = [(float(x), float(y)) for x, y in vertices_xy[face]]
            shade = _xhand_face_shade(xhand_pose.vertices[face], xhand_pose.side)
            draw.polygon(polygon, fill=(shade, shade, shade, 54), outline=(90, 90, 90, 28))

    tactile_xy, _ = to_screen(xhand_pose.taxel_points)
    colors = _value_colors(xhand_pose.tactile_values)
    for xy, color in zip(tactile_xy, colors):
        radius = 2
        draw.ellipse(
            (float(xy[0] - radius), float(xy[1] - radius), float(xy[0] + radius), float(xy[1] + radius)),
            fill=tuple(color) + (235,),
            outline=(255, 255, 255, 180),
        )

    tactile = xhand_pose.tactile_values
    draw.text(
        (14, 12),
        title or f"{xhand_pose.side} XHand tactile map at real qpos",
        fill=(20, 20, 20, 255),
    )
    draw.text(
        (14, height - 30),
        f"taxels={tactile.shape[0]} mesh_vertices={xhand_pose.vertices.shape[0]} "
        f"frame={xhand_pose.coordinate_frame} "
        f"tactile min/mean/max={float(tactile.min()):.3g}/{float(tactile.mean()):.3g}/{float(tactile.max()):.3g}",
        fill=(75, 75, 75, 255),
    )
    image.convert("RGB").save(out_path)


def _render_mano_xhand_overlay(
    result: Any,
    xhand_pose: XHandTactilePose,
    out_path: str | Path,
    *,
    width: int = 900,
    height: int = 700,
    title: str | None = None,
) -> None:
    from PIL import Image, ImageDraw

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    margin = 44
    bounds_parts = [
        result.mano_mesh.vertices,
        result.taxel_surface_points,
        xhand_pose.taxel_points,
    ]
    if xhand_pose.vertices.size:
        bounds_parts.append(xhand_pose.vertices[::_mesh_stride(xhand_pose.vertices.shape[0], 60000)])
    bounds_points = np.concatenate(bounds_parts, axis=0)
    projected_bounds, _ = _project_xhand_view(bounds_points, xhand_pose.side)
    xy_min = projected_bounds.min(axis=0)
    xy_max = projected_bounds.max(axis=0)
    extent = np.maximum(xy_max - xy_min, 1e-9)
    scale = min((width - 2 * margin) / extent[0], (height - 2 * margin) / extent[1])
    center = (xy_min + xy_max) / 2.0

    def to_screen(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy, z = _project_xhand_view(points, xhand_pose.side)
        screen = np.empty_like(xy)
        screen[:, 0] = (xy[:, 0] - center[0]) * scale + width / 2.0
        screen[:, 1] = height / 2.0 - (xy[:, 1] - center[1]) * scale
        return screen, z

    if xhand_pose.vertices.size and xhand_pose.faces.size:
        xhand_xy, xhand_depth = to_screen(xhand_pose.vertices)
        face_step = _mesh_stride(xhand_pose.faces.shape[0], 22000)
        faces = xhand_pose.faces[::face_step]
        face_depth = xhand_depth[faces].mean(axis=1)
        for face_idx in np.argsort(face_depth):
            face = faces[int(face_idx)]
            polygon = [(float(x), float(y)) for x, y in xhand_xy[face]]
            draw.polygon(polygon, fill=(88, 88, 88, 20), outline=(60, 60, 60, 30))

    mano_xy, mano_depth = to_screen(result.mano_mesh.vertices)
    mano_face_depth = mano_depth[result.mano_mesh.faces].mean(axis=1)
    for face_idx in np.argsort(mano_face_depth):
        face = result.mano_mesh.faces[int(face_idx)]
        polygon = [(float(x), float(y)) for x, y in mano_xy[face]]
        draw.polygon(polygon, fill=(90, 138, 205, 70), outline=(35, 76, 135, 55))

    colors = _value_colors(xhand_pose.tactile_values)
    xhand_taxel_xy, _ = to_screen(xhand_pose.taxel_points)
    for xy, color in zip(xhand_taxel_xy, colors):
        radius = 2
        draw.ellipse(
            (float(xy[0] - radius), float(xy[1] - radius), float(xy[0] + radius), float(xy[1] + radius)),
            fill=tuple(color) + (230,),
            outline=(255, 255, 255, 170),
        )

    mano_taxel_xy, _ = to_screen(result.taxel_surface_points)
    for xy in mano_taxel_xy[::_mesh_stride(mano_taxel_xy.shape[0], 180)]:
        radius = 1
        draw.ellipse(
            (float(xy[0] - radius), float(xy[1] - radius), float(xy[0] + radius), float(xy[1] + radius)),
            fill=(35, 76, 135, 170),
        )

    draw.text((14, 12), title or "MANO/XHand overlay in MANO frame", fill=(20, 20, 20, 255))
    draw.text(
        (14, height - 30),
        "XHand grey wire + colored tactile; MANO translucent blue; shared MANO-frame wrist transform",
        fill=(75, 75, 75, 255),
    )
    image.convert("RGB").save(out_path)


def _write_side_by_side(image_paths: Sequence[Path], labels: Sequence[str], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    if len(image_paths) != len(labels):
        raise ValueError("image_paths and labels must have the same length")
    label_height = 28
    loaded = [Image.open(path).convert("RGB") for path in image_paths]
    panel_height = max(image.height for image in loaded)
    width = sum(image.width for image in loaded)
    image = Image.new("RGB", (width, panel_height + label_height), "white")
    draw = ImageDraw.Draw(image)
    x = 0
    for label, panel in zip(labels, loaded):
        draw.text((x + 12, 8), label, fill=(20, 20, 20))
        image.paste(panel, (x, label_height))
        x += panel.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _project_xhand_view(points: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    view_sign = -1.0 if side == "left" else 1.0
    x = points[:, 0] + 0.85 * view_sign * points[:, 1]
    y = points[:, 2] - 0.12 * view_sign * points[:, 1]
    depth = view_sign * points[:, 1]
    return np.stack([x, y], axis=1), depth


def _xhand_face_shade(triangle: np.ndarray, side: str) -> int:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-18:
        return 188
    normal /= norm
    view_sign = -1.0 if side == "left" else 1.0
    light = np.asarray([0.35, 0.25 * view_sign, 0.90], dtype=np.float64)
    light /= np.linalg.norm(light)
    value = 0.55 + 0.45 * abs(float(np.dot(normal, light)))
    return int(165 + 55 * value)


def _value_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        scaled = np.zeros_like(values)
    else:
        low = float(finite.min())
        high = float(finite.max())
        if abs(high - low) <= 1e-18:
            scaled = np.full_like(values, 0.5, dtype=np.float64)
        else:
            scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    blue = np.asarray([39, 118, 214], dtype=np.float64)
    yellow = np.asarray([244, 196, 48], dtype=np.float64)
    red = np.asarray([225, 78, 54], dtype=np.float64)
    colors = np.empty((values.shape[0], 3), dtype=np.float64)
    lower = scaled <= 0.5
    colors[lower] = blue + (yellow - blue) * (scaled[lower, None] * 2.0)
    colors[~lower] = yellow + (red - yellow) * ((scaled[~lower, None] - 0.5) * 2.0)
    return np.clip(colors, 0, 255).astype(np.uint8)


def _mesh_stride(count: int, max_items: int) -> int:
    return max(1, int(ceil(count / max_items)))


def _axis_angle_matrix(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    theta = float(np.linalg.norm(vector))
    if theta <= 1e-12:
        return np.eye(3, dtype=np.float64) + _skew(vector)
    axis = vector / theta
    skew = _skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _rot_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_angle(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)


def _write_montage(png_paths: Sequence[Path], labels: Sequence[str], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    if not png_paths:
        return
    tile_width = 420
    label_height = 26
    loaded = [Image.open(path).convert("RGB") for path in png_paths]
    resized = []
    for image in loaded:
        scale = tile_width / float(image.width)
        resized.append(image.resize((tile_width, int(round(image.height * scale)))))
    tile_height = max(image.height for image in resized)
    montage = Image.new("RGB", (tile_width * len(resized), tile_height + label_height), "white")
    draw = ImageDraw.Draw(montage)
    for idx, image in enumerate(resized):
        x0 = idx * tile_width
        montage.paste(image, (x0, label_height))
        draw.text((x0 + 10, 7), labels[idx], fill=(20, 20, 20))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


if __name__ == "__main__":
    main()
