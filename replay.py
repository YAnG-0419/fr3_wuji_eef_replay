#!/usr/bin/env python3
"""Replay LeRobot cam0/link8 EEF episodes through the existing FR3/Wuji stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import yaml

from core import Kinematics, decode, digest, joint_names, read_episode, read_limits, save_bundle, trajectory

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "Convert_data/outputs/fr3_wuji/tomato_eef_cam0"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Offline EEF -> base frame -> IK -> bounded trajectory; no devices")
    plan.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    plan.add_argument("--episode", type=int, default=0)
    plan.add_argument("--field", choices=("action", "observation.state"), default="action")
    plan.add_argument("--start-frame", type=int, default=0)
    plan.add_argument("--end-frame", type=int)
    plan.add_argument("--calibration", type=Path, help="Default: dataset/conversion/calibration.yaml")
    plan.add_argument("--limits", type=Path, default=HERE / "limits.yaml")
    plan.add_argument("--seed-mode", choices=("recorded", "previous"), default="recorded")
    plan.add_argument("--interpolation", choices=("bounded", "interpolate"), default="bounded",
                      help="bounded smooths within the joint control hull; interpolate passes every IK knot")
    plan.add_argument("--max-ik-step", type=float, default=0.3, help="Maximum adjacent arm joint delta, radians")
    plan.add_argument("--speed", type=float, default=0.4, help="Requested playback speed in (0,1]; may slow further")
    plan.add_argument("--output", type=Path)
    run = sub.add_parser("run", help="Use installed openpi device runtime; default only renders launch commands")
    run.add_argument("--plan", type=Path, required=True, help="Directory containing report.json and trajectory.npz")
    run.add_argument("--openpi", type=Path, default=HERE.parent / "openpi")
    run.add_argument("--reference", type=Path, default=Path("/home/user/lpy/gello-retarget"))
    run.add_argument("--sdk-site", type=Path, default=Path("/home/user/miniconda3/envs/gello-upper-body-teleop/lib/python3.10/site-packages"))
    run.add_argument("--overlay", type=Path, help="Default: OPENPI/.deployment/oneshot-overlay")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Start real device processes and send motion")
    mode.add_argument("--read-only", action="store_true", help="Read feedback only; existing arm publishers required")
    admission = run.add_mutually_exclusive_group()
    admission.add_argument("--supervised-trial", action="store_true")
    admission.add_argument("--qualification", type=Path)
    run.add_argument("--left-arm-ip", default="172.16.0.2")
    run.add_argument("--right-arm-ip", default="172.16.1.2")
    run.add_argument("--wuji-left-address", default="192.168.1.110:7447")
    run.add_argument("--wuji-right-address", default="192.168.2.111:7447")
    run.add_argument("--hand-control", choices=("strict", "slider"), default="strict")
    run.add_argument("--max-start-delta", type=float, default=0.3, help="Arm first-pose admission bound, radians")
    run.add_argument("--settle-timeout", type=float, default=15)
    run.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.execute and not (args.supervised_trial or args.qualification):
            parser.error("--execute requires --supervised-trial or --qualification, as in the openpi runtime")
        if not args.execute and (args.supervised_trial or args.qualification):
            parser.error("Execution admission flags require --execute")
        from live import run as run_live
        return run_live(args)
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("--speed must be in (0,1]")
    calibration_path = args.calibration or args.dataset / "conversion/calibration.yaml"
    calibration = yaml.safe_load(calibration_path.read_text())
    model = Kinematics(calibration)
    lower, upper = read_limits(args.limits)
    eef, seeds, times, frames, parquet = read_episode(
        args.dataset, args.episode, args.field, args.start_frame, args.end_frame
    )
    print(f"【离线】episode={args.episode}, {len(times)}帧；校验坐标变换和双臂 IK…", flush=True)
    positions, targets, errors = decode(eef, seeds, model, lower, upper, args.seed_mode, args.max_ik_step)
    playback, motion = trajectory(times, positions, lower, upper, 1 / args.speed,
                                  interpolation=args.interpolation)
    smoothed = playback.sample(times * playback.scale)
    deviation = np.empty_like(errors)
    for i in range(len(times)):
        for j, section in enumerate((slice(0, 7), slice(27, 34))):
            error = model.error(model.fk(smoothed[i, section]), targets[i, j])
            deviation[i, j] = [np.linalg.norm(error[:3]), np.linalg.norm(error[3:])]
    motion["max_eef_position_knot_deviation_m"] = float(deviation[:, :, 0].max())
    motion["max_eef_rotation_knot_deviation_rad"] = float(deviation[:, :, 1].max())
    output = args.output or HERE / "outputs" / f"episode-{args.episode:06d}-{uuid.uuid4().hex[:8]}"
    report = {
        "schema": "fr3-wuji-eef-replay-v1", "hardware_output": False,
        "dataset": str(args.dataset.resolve()), "episode": args.episode, "field": args.field,
        "start_frame": int(frames[0]), "end_frame_exclusive": int(frames[-1] + 1), "frames": len(times),
        "joint_names": joint_names(), "calibration": calibration,
        "source_parquet": str(parquet), "source_parquet_sha256": digest(parquet),
        "calibration_sha256": digest(calibration_path), "limits_sha256": digest(args.limits),
        "seed_mode": args.seed_mode, "requested_speed": args.speed, "minimum_time_scale": 1 / args.speed,
        "interpolation": args.interpolation,
        "max_ik_step_rad": args.max_ik_step,
        "max_fk_position_error_m": float(errors[:, :, 0].max()),
        "max_fk_rotation_error_rad": float(errors[:, :, 1].max()),
        "source_seconds": float(times[-1]), "playback": motion,
        "trajectory_note": "Joint quintic B-spline of verified EEF/IK knots. Bounded mode smooths; interpolate mode passes knots. No Cartesian line or collision guarantee.",
        "engaged_note": "Replay follows all selected targets, including disengaged intervals; no engagement gating.",
    }
    save_bundle(output, {"time": times, "positions": positions, "eef": eef,
                         "base_link8": targets, "fk_errors": errors, "frame_index": frames,
                         "lower": lower, "upper": upper}, report)
    print(json.dumps({"output": str(output.resolve()), "hardware_output": False,
                      "frames": len(times), "source_seconds": float(times[-1]),
                      "playback_seconds": playback.duration,
                      "smoothed_eef_max_knot_deviation_m": motion["max_eef_position_knot_deviation_m"],
                      "max_fk_position_error_m": report["max_fk_position_error_m"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        print(f"回放工具：{error}", file=sys.stderr)
        raise SystemExit(1)
