"""Model-free replay adapter for openpi's existing bounded device bridge.

Imports device/runtime code only after an explicit `run` subcommand. Offline
records are never relabelled as fresh model inference or live measurements.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import fcntl
import hashlib
import ipaddress
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

import numpy as np
import yaml

from core import ARM, HAND, check_bounds, digest, joint_names, load_bundle, positive, trajectory


def stream(devices, playback, approach, start, plan_hash, frame_type, *,
           stop_requested, health_check=lambda: None, settle_timeout=15,
           clock=time.monotonic, sleep=time.sleep):
    """Finite 100Hz stream; deadlines belong to generated frames, not the dataset."""
    positive(settle_timeout)
    run_id = uuid.uuid4().hex
    player = SimpleNamespace(run_id=run_id, digest=plan_hash, plan=SimpleNamespace(start=start))
    # Allocate before enabling devices. No disk IO or planning in the stream.
    duration = approach.duration + playback.duration
    capacity = int(np.ceil((duration + settle_timeout + 1) * 100)) + 2
    history_t = np.empty(capacity)
    commands = np.empty((capacity, 54))
    feedback_q = np.empty((capacity, 54))
    devices.prepare(player, None, clock())
    began = clock()
    sequence, stable_since = 0, None
    endpoint = playback.sample(playback.duration)
    try:
        while True:
            if stop_requested():
                raise InterruptedError("Operator requested replay stop")
            health_check()
            due = began + sequence * 0.01
            now = clock()
            if now < due:
                sleep(due - now)
            now = clock()
            if now - due > 0.01:
                raise RuntimeError("100Hz replay deadline missed; no burst catch-up")
            elapsed = now - began
            if sequence == 0:
                # First output is exactly the measured acquisition pose.
                q, dq, phase = start.copy(), np.zeros(54), "approach"
            elif elapsed < approach.duration:
                q, dq, phase = approach.sample(elapsed), approach.sample(elapsed, 1), "approach"
            elif elapsed < duration:
                t = elapsed - approach.duration
                q, dq, phase = playback.sample(t), playback.sample(t, 1), "playback"
            else:
                q, dq, phase = endpoint, np.zeros(54), "settle_end"
            frame = frame_type(run_id=run_id, plan_hash=plan_hash, sequence=sequence,
                               created=now, valid_until=now + 0.02, plan_elapsed=elapsed,
                               positions=q, velocities=dq, phase=phase)
            feedback = devices.submit(frame, clock())
            if feedback is None:
                raise RuntimeError("Device runtime must acknowledge each frame with measured feedback")
            feedback.check(clock(), feedback.epoch)
            history_t[sequence] = elapsed
            commands[sequence] = q
            feedback_q[sequence] = feedback.positions
            sequence += 1
            if phase == "settle_end":
                checked = ARM if feedback.hand_control == "slider" else np.arange(54)
                stationary = np.max(abs(feedback.velocities[checked])) <= 0.02
                reached = (np.max(abs(feedback.positions[ARM] - endpoint[ARM])) <= 0.03
                           and feedback.hand_targets_reached
                           and feedback.hands_settled(endpoint, 0.1))
                stable_since = (now if stable_since is None else stable_since) if stationary and reached else None
                if stable_since is not None and now - stable_since >= 0.5:
                    result = devices.finish()
                    return result, {"time": history_t[:sequence], "command": commands[:sequence],
                                    "feedback": feedback_q[:sequence]}
                if elapsed > duration + settle_timeout:
                    raise RuntimeError("Endpoint settling timed out")
            if sequence >= capacity:
                raise RuntimeError("Replay exceeded its bounded recording allocation")
    except BaseException:
        # Disconnect is the fallback if an RPC timed out; the bridge owns its watchdog.
        try:
            devices.stop()
        finally:
            devices.close()
        raise


def runtime(args, hardware, output, report):
    """Use the same build verifier, reference limits, ownership lock and launchers."""
    from experiments.weight_motion_eval.reference import deployment_module
    module = deployment_module("record_inference")
    limits = module.load_limits(reference=args.reference)
    names = list(module.NAMES)
    if names != joint_names():
        raise ValueError("Installed openpi device order does not match replay order")
    manifest = hardware.verify_overlay()
    # Launch helper still uses the verified overlay; do not modify it or the reference.
    workcell = yaml.safe_load((args.reference / "config/workcell/current.yaml").read_text())
    for side, ip in (("LEFT", args.left_arm_ip), ("RIGHT", args.right_arm_ip)):
        if str(workcell[side]["use_fake_hardware"]).lower() != "false":
            raise ValueError("Expected a real workcell profile")
        workcell[side]["robot_ip"] = ip
    (output / "workcell.yaml").write_text(yaml.safe_dump(workcell))
    config = {
        "schema_version": 1, "reference": str(args.reference),
        "hands": {"left": args.wuji_left_address, "right": args.wuji_right_address},
        "arms": {"left": args.left_arm_ip, "right": args.right_arm_ip},
        "limits": {key: limits[key].tolist() for key in ("lower", "upper")},
        "reference_hashes": limits["sources"], "joint_names": names,
        "controller_library": manifest["built_library"],
        "controller_sha256": manifest["built_library_sha256"],
        "checkpoint": {"source": "offline_eef_replay", "episode": report["episode"]},
        "finish_policy": "disable", "hand_control": args.hand_control,
        "continuous": False, "arm_speed_rad_s": 0.7,
    }
    (output / "runtime.json").write_text(json.dumps(config, indent=2) + "\n")
    return limits, config


def run(args):
    for name in ("openpi", "reference", "sdk_site"):
        setattr(args, name, getattr(args, name).resolve())
    positive(args.max_start_delta)
    positive(args.settle_timeout)
    for value in (args.left_arm_ip, args.right_arm_ip):
        ipaddress.IPv4Address(value)
    for value in (args.wuji_left_address, args.wuji_right_address):
        address, port = value.rsplit(":", 1)
        ipaddress.IPv4Address(address)
        if not 1 <= int(port) <= 65535:
            raise ValueError("Invalid hand port")
    if args.left_arm_ip == args.right_arm_ip or args.wuji_left_address == args.wuji_right_address:
        raise ValueError("Left/right device addresses must be distinct")
    data, report, playback = load_bundle(args.plan)
    sys.path.insert(0, str(args.openpi))
    from deploy.fr3_wuji_runtime import hardware
    from deploy.fr3_wuji_slow.core import Frame
    from deploy.fr3_wuji_slow.ipc import RemoteDevices

    # Runtime-local path configuration; no edits to the user's deployment repository.
    hardware.REFERENCE = args.reference
    hardware.SDK = args.sdk_site
    hardware.OVERLAY = (args.overlay or args.openpi / ".deployment/oneshot-overlay").resolve()
    args.finish_policy, args.continuous = "disable", False
    output = (args.output or Path(__file__).parent / "outputs" / ("live-" + uuid.uuid4().hex[:10])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    limits, runtime_config = runtime(args, hardware, output, report)
    # Replan against the actual machine's limits; bundled limits cannot relax them.
    lower = np.maximum(limits["lower"], data["lower"])
    upper = np.minimum(limits["upper"], data["upper"])
    if np.any(lower >= upper):
        raise ValueError("Installed and bundled joint limits have no common range")
    check_bounds(data["positions"], lower, upper)
    playback, _ = trajectory(data["time"], data["positions"], lower, upper, report["minimum_time_scale"],
                             interpolation=report["interpolation"])
    (output / "source-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not args.execute and not args.read_only:
        preview = copy.copy(args)
        preview.execute = True
        preview.supervised_trial = True
        commands = hardware.launch_commands(preview, output, Path("/tmp/eef-replay-PREVIEW"))
        (output / "commands-preview.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(f"【检查】配置已生成：{output}；未启动设备。")
        return
    hardware.preflight_owners(execute=args.execute)
    qualification = None
    if args.execute and args.qualification:
        from experiments.weight_motion_eval.oneshot.qualification import Qualification
        qualification = Qualification(args.qualification, controller_sha256=runtime_config["controller_sha256"],
                                      hand_control=args.hand_control)
    stopped = False

    def request_stop(*_):
        nonlocal stopped
        stopped = True

    with ExitStack() as stack:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(sig, request_stop)
            stack.callback(signal.signal, sig, previous)
        lock = stack.enter_context((args.openpi / ".deployment/oneshot-owner.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ipc = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="eef-replay-")))
        children = hardware.Children(output)
        stack.callback(children.close)
        commands = hardware.launch_commands(args, output, ipc)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        for label, command in commands.items():
            children.launch(label, command)
        deadline = time.monotonic() + 45
        while not (ipc / "devices.sock").exists():
            children.check()
            if stopped or time.monotonic() > deadline:
                raise RuntimeError("Device startup stopped or timed out")
            time.sleep(0.05)
        devices = RemoteDevices(ipc / "devices.sock", execute=args.execute, finish_policy="disable")
        stack.callback(devices.close)
        if args.execute:
            stack.callback(devices.stop)
        feedback = None
        while time.monotonic() < deadline and not stopped:
            children.check()
            try:
                feedback = devices.feedback(time.monotonic())
                break
            except RuntimeError:
                time.sleep(0.02)
        if feedback is None:
            raise RuntimeError("No fresh four-device feedback; inspect bridge logs")
        if args.read_only:
            (output / "feedback.json").write_text(json.dumps({
                "hardware_output": False, "positions": feedback.positions.tolist(),
                "velocities": feedback.velocities.tolist(), "joint_names": joint_names()
            }, indent=2) + "\n")
            print(f"【只读】已读取双臂双手状态：{output}；未使能电机。")
            return
        delta = np.max(abs(data["positions"][0, ARM] - feedback.positions[ARM]))
        if delta > args.max_start_delta:
            raise ValueError(f"First arm pose delta {delta:.4f}rad exceeds --max-start-delta {args.max_start_delta}; "
                             "move near the recorded start pose before replay")
        if qualification is not None:
            hand_delta = np.max(abs(data["positions"][0, HAND] - feedback.positions[HAND]))
            if hand_delta > qualification.data["hand_raw_initial_delta_rad"]:
                raise ValueError("First hand pose exceeds commissioning acquisition bound")
        checked = ARM if args.hand_control == "slider" else np.arange(54)
        if np.max(abs(feedback.velocities[checked])) > 0.02:
            raise ValueError("Devices must be stationary before replay acquisition")
        approach, approach_report = trajectory(np.array([0., 1.]),
            np.stack([feedback.positions, data["positions"][0]]), lower, upper, approach=True)
        # Bind this run to the complete offline bundle AND the actual measured start.
        h = hashlib.sha256()
        h.update(digest(args.plan / "report.json").encode())
        h.update(digest(args.plan / "trajectory.npz").encode())
        h.update(np.asarray(feedback.positions, dtype="<f8").tobytes())
        (output / "approach.json").write_text(json.dumps(approach_report, indent=2) + "\n")
        print(f"【真机】接近 {approach.duration:.1f}s，回放 {playback.duration:.1f}s；Ctrl-C 停止。", flush=True)
        # Planning can be slow; prepare() compares this start with fresh device
        # feedback before enabling and again after acquisition.
        from experiments.weight_motion_eval.oneshot.motion_gc import MotionGC
        motion_gc = MotionGC()
        try:
            motion_gc.begin()
            result, history = stream(devices, playback, approach, feedback.positions, h.hexdigest(), Frame,
                stop_requested=lambda: stopped, health_check=children.check, settle_timeout=args.settle_timeout)
        except BaseException as error:
            (output / "result.json").write_text(json.dumps({"completed": False, "error": str(error),
                "physical_stop_confirmed": False}, ensure_ascii=False, indent=2) + "\n")
            raise
        finally:
            motion_gc.end()
        np.savez_compressed(output / "execution.npz", **history)
        (output / "result.json").write_text(json.dumps({"completed": True, **result}, indent=2) + "\n")
        print(f"【完成】轨迹结束，双手按现有部署逻辑释放。记录：{output}")
