"""Offline EEF decoding, FR3 IK and bounded joint interpolation. No device imports."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import BSpline, PPoly, make_interp_spline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

ARM = np.r_[0:7, 27:34]
HAND = np.r_[7:27, 34:54]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def joint_names():
    result = []
    for side, prefix in (("left", "l"), ("right", "r")):
        result.extend(f"{side}_fr3_joint{i}" for i in range(1, 8))
        suffixes = ["thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip"]
        for finger in ("index_finger", "middle_finger", "ring_finger", "pinky"):
            suffixes.extend(f"{finger}_{j}" for j in ("mcp_flex", "mcp_abd", "pip", "dip"))
        result.extend(f"{prefix}_{s}" for s in suffixes)
    return result


def array(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{label}: expected finite {shape}, got {a.shape}")
    return a


def positive(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Expected a positive finite number")
    return value


def pose_matrix(pose):
    pose = array(pose, (7,), "xyz + quaternion xyzw")
    norm = np.linalg.norm(pose[3:])
    if abs(norm - 1) > 1e-3:
        raise ValueError(f"Quaternion norm {norm:g} is not approximately one")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:] / norm).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


class Kinematics:
    def __init__(self, calibration):
        if calibration.get("schema") != "fr3-cam0-handeye-v1":
            raise ValueError("Expected fr3-cam0-handeye-v1 calibration")
        if calibration.get("eef", {}).get("definition") != "fr3_link8":
            raise ValueError("Only FR3 link8 EEF is supported; no implicit TCP offset")
        convention = calibration["transform_convention"]
        if (convention["stored_transform"], convention["quaternion_order"], convention["translation_unit"]) != (
            "base_to_camera", "xyzw", "meter"
        ):
            raise ValueError("Unexpected calibration transform convention")
        self.origins, self.types = [], []
        for joint in calibration["fr3_link0_to_link8_kinematics"]["joints"]:
            origin = np.eye(4)
            origin[:3, :3] = Rotation.from_euler("xyz", joint["origin_rpy_rad"]).as_matrix()
            origin[:3, 3] = array(joint["origin_xyz_m"], (3,), "joint origin")
            kind = joint["type"]
            if kind not in {"fixed", "revolute"} or (
                kind == "revolute" and not np.array_equal(joint["axis"], [0, 0, 1])
            ):
                raise ValueError("Expected FR3 fixed joints / revolute Z axes")
            self.origins.append(origin)
            self.types.append(kind)
        if self.types.count("revolute") != 7 or self.types[-1] != "fixed":
            raise ValueError("Expected seven FR3 joints and fixed link8")
        self.base_camera = {}
        for side in ("left", "right"):
            arm = calibration["arms"][side]
            if arm["base_frame"] != f"{side}_fr3_link0" or arm["eef_frame"] != f"{side}_fr3_link8":
                raise ValueError("Unexpected arm base / EEF frame")
            extrinsic = arm["base_to_camera"]
            self.base_camera[side] = pose_matrix([
                *extrinsic["translation_xyz_m"], *extrinsic["quaternion_xyzw"]
            ])

    def fk(self, q):
        q = array(q, (7,), "FR3 joint angles")
        t, j = np.eye(4), 0
        for origin, kind in zip(self.origins, self.types):
            t = t @ origin
            if kind == "revolute":
                c, s = np.cos(q[j]), np.sin(q[j])
                rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                t = t @ rz
                j += 1
        return t

    @staticmethod
    def error(actual, target):
        return np.r_[actual[:3, 3] - target[:3, 3],
                     Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()]

    def ik(self, target, seed, lower, upper, position_tol=1e-5, rotation_tol=1e-4):
        seed = array(seed, (7,), "IK seed")
        lower, upper = np.asarray(lower), np.asarray(upper)
        if np.any(seed < lower) or np.any(seed > upper):
            raise ValueError("Recorded IK seed exceeds arm joint limits")
        # This dataset preserves the original joints. They resolve the redundant
        # 7-DoF branch, but the EEF target is always checked and remains authoritative.
        error = self.error(self.fk(seed), target)
        if np.linalg.norm(error[:3]) <= position_tol and np.linalg.norm(error[3:]) <= rotation_tol:
            return seed.copy(), error

        def residual(q):
            return np.r_[self.error(self.fk(q), target), 1e-5 * (q - seed)]

        solved = least_squares(residual, seed, bounds=(lower, upper), max_nfev=200,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        error = self.error(self.fk(solved.x), target)
        if np.linalg.norm(error[:3]) > position_tol or np.linalg.norm(error[3:]) > rotation_tol:
            raise ValueError(f"IK failed: position={np.linalg.norm(error[:3]):.6g}m, "
                             f"rotation={np.linalg.norm(error[3:]):.6g}rad")
        return solved.x, error


def read_limits(path):
    document = yaml.safe_load(Path(path).read_text())
    if document.get("joint_names") != joint_names():
        raise ValueError("Limits joint names/order do not match replay contract")
    lower = array(document["lower"], (54,), "lower limits")
    upper = array(document["upper"], (54,), "upper limits")
    if np.any(lower >= upper):
        raise ValueError("Invalid joint limits")
    return lower, upper


def read_episode(root, episode, field="action", start=0, end=None):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if field not in {"action", "observation.state"}:
        raise ValueError("Replay field must be action or observation.state")
    if not 0 <= episode < info["total_episodes"]:
        raise ValueError("Episode index outside dataset")
    expected = [f"{side}_link8.cam0.{component}" for side in ("left", "right")
                for component in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    expected += [n + (".command" if field == "action" else ".position")
                 for n in np.asarray(joint_names())[HAND]]
    if info["features"][field]["names"] != expected or info["features"][field]["shape"] != [54]:
        raise ValueError("Dataset is not the supported 54D cam0/link8 EEF layout")
    path = root / info["data_path"].format(episode_index=episode,
                                         episode_chunk=episode // info["chunks_size"])
    seed_field = "action_joint" if field == "action" else "observation.joint_state"
    table = pq.read_table(path, columns=[field, seed_field, "timestamp", "frame_index", "episode_index"])
    end = len(table) if end is None else end
    if not 0 <= start < end <= len(table) or end - start < 2:
        raise ValueError("Select at least two frames; end-frame is exclusive")
    table = table.slice(start, end - start)
    eef = array(table[field].to_pylist(), (len(table), 54), field)
    times = array(table["timestamp"].to_numpy(), (len(table),), "timestamp")
    if np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > 0.2:
        raise ValueError("Non-monotonic timestamps or a gap greater than 0.2s")
    frames = table["frame_index"].to_numpy()
    if not np.array_equal(frames, np.arange(start, end)) or not np.all(table["episode_index"].to_numpy() == episode):
        raise ValueError("Unexpected frame / episode ordering")
    seeds = np.asarray(table[seed_field].to_pylist(), dtype=float)
    # joint_state contains position AND velocity names; use explicit position names.
    original_names = info["features"][seed_field]["names"]
    indices = [original_names.index(name + ".position") for name in np.asarray(joint_names())[ARM]]
    arm_seed = array(seeds[:, indices], (len(table), 14), "recorded arm seeds")
    return eef, arm_seed, times - times[0], frames, path


def decode(eef, seeds, model, lower, upper, seed_mode="recorded", max_step=0.3):
    if seed_mode not in {"recorded", "previous"}:
        raise ValueError("Invalid IK seed mode")
    positive(max_step)
    q = np.empty((len(eef), 54))
    q[:, HAND] = eef[:, 14:54]
    targets = np.empty((len(eef), 2, 4, 4))
    errors = np.empty((len(eef), 2, 2))
    for i in range(len(eef)):
        for j, (side, section) in enumerate((("left", slice(0, 7)), ("right", slice(27, 34)))):
            target = model.base_camera[side] @ pose_matrix(eef[i, j * 7:(j + 1) * 7])
            seed = seeds[i, j * 7:(j + 1) * 7] if seed_mode == "recorded" or i == 0 else q[i - 1, section]
            try:
                q[i, section], error = model.ik(target, seed, lower[section], upper[section])
            except ValueError as exc:
                raise ValueError(f"Frame offset {i}, {side}: {exc}") from exc
            targets[i, j] = target
            errors[i, j] = [np.linalg.norm(error[:3]), np.linalg.norm(error[3:])]
        if i and np.max(np.abs(q[i, ARM] - q[i - 1, ARM])) > max_step:
            raise ValueError(f"Frame offset {i}: arm IK branch jump > {max_step} rad")
    check_bounds(q, lower, upper)
    return q, targets, errors


def check_bounds(q, lower, upper):
    if not np.isfinite(q).all():
        raise ValueError("Non-finite joint command")
    bad = np.argwhere((q < lower) | (q > upper))
    if len(bad):
        row, joint = bad[0]
        raise ValueError(f"Joint limit: frame {row}, {joint_names()[joint]}={q[row, joint]:g}, "
                         f"allowed [{lower[joint]:g}, {upper[joint]:g}]; no automatic clipping")


def extrema(spline):
    low, high = np.empty((4, 54)), np.empty((4, 54))
    start, end = spline.t[spline.k], spline.t[-spline.k - 1]
    for joint in range(54):
        poly = PPoly.from_spline((spline.t, spline.c[:, joint], spline.k))
        knots = poly.x[(poly.x >= start) & (poly.x <= end)]
        for order in range(4):
            curve = poly.derivative(order)
            roots = curve.derivative().roots(extrapolate=False)
            roots = roots[np.isfinite(roots) & (roots >= start) & (roots <= end)]
            values = curve(np.r_[knots, roots])
            low[order, joint], high[order, joint] = values.min(), values.max()
    return low, high


def motion_caps(approach=False):
    caps = np.empty((3, 54))
    caps[:, ARM] = np.array([0.18, 0.432, 3.456] if approach else [0.42, 1.44, 17.28])[:, None]
    caps[:, HAND] = np.array([0.35, 1.0, 5.0] if approach else [np.pi / 4, 3.0, 30.0])[:, None]
    return caps


@dataclass
class Trajectory:
    spline: BSpline
    scale: float

    @property
    def duration(self):
        return float(self.spline.t[-self.spline.k - 1] * self.scale)

    def sample(self, elapsed, derivative=0):
        return self.spline(np.clip(elapsed / self.scale, 0, self.duration / self.scale), nu=derivative) / self.scale**derivative


def trajectory(times, positions, lower, upper, minimum_scale=1, approach=False, interpolation="bounded"):
    positive(minimum_scale)
    times = array(times, (len(positions),), "trajectory times")
    positions = array(positions, (len(times), 54), "joint trajectory")
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory needs increasing timestamps starting at zero")
    if interpolation == "bounded":
        # Positive B-spline basis: every point stays in the convex hull of its
        # nearby recorded targets. Tripled endpoint controls give zero endpoint
        # velocity and acceleration. This smooths knots rather than interpolating
        # them exactly, and never clips a resulting command at joint limits.
        knots = np.r_[np.repeat(0., 6), times[1:-1], np.repeat(times[-1], 6)]
        controls = np.concatenate([np.repeat(positions[:1], 3, axis=0), positions[1:-1],
                                   np.repeat(positions[-1:], 3, axis=0)])
        spline = BSpline(knots, controls, 5, extrapolate=False)
    elif interpolation == "interpolate":
        zeros = np.zeros(54)
        spline = make_interp_spline(times, positions, k=5,
                                   bc_type=([(1, zeros), (2, zeros)], [(1, zeros), (2, zeros)]))
    else:
        raise ValueError("Unknown interpolation mode")
    low, high = extrema(spline)
    # Polynomial conversion/root finding introduces a few ulps at exact limits.
    if np.any(low[0] < lower - 1e-9) or np.any(high[0] > upper + 1e-9):
        raise ValueError("Continuous interpolation overshoots joint limits; use bounded interpolation")
    peak = np.maximum(abs(low[1:]), abs(high[1:]))
    ratios = peak / motion_caps(approach)
    scale = max(minimum_scale, *(float(np.max(ratios[i])) ** (1 / (i + 1)) for i in range(3)))
    result = Trajectory(spline, scale * (1 + 1e-8))
    return result, {"seconds": result.duration, "time_scale": result.scale, "interpolation": interpolation,
                    "max_joint_knot_deviation_rad": float(np.max(abs(spline(times) - positions))),
                    "peak_velocity_rad_s": float(np.max(peak[0] / result.scale)),
                    "peak_acceleration_rad_s2": float(np.max(peak[1] / result.scale**2)),
                    "peak_jerk_rad_s3": float(np.max(peak[2] / result.scale**3))}


def save_bundle(output, arrays, report):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectory.npz", **arrays)
    report["trajectory_sha256"] = digest(output / "trajectory.npz")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_bundle(path):
    path = Path(path).resolve()
    report = json.loads((path / "report.json").read_text())
    if report.get("schema") != "fr3-wuji-eef-replay-v1" or report["joint_names"] != joint_names():
        raise ValueError("Unsupported replay bundle")
    if digest(path / "trajectory.npz") != report["trajectory_sha256"]:
        raise ValueError("Replay NPZ hash mismatch")
    with np.load(path / "trajectory.npz", allow_pickle=False) as f:
        data = {key: f[key].copy() for key in f.files}
    n = len(data["time"])
    data["positions"] = array(data["positions"], (n, 54), "saved joint trajectory")
    lower = array(data["lower"], (54,), "saved lower limits")
    upper = array(data["upper"], (54,), "saved upper limits")
    if np.any(lower >= upper):
        raise ValueError("Invalid saved joint limits")
    model = Kinematics(report["calibration"])
    eef = array(data["eef"], (n, 54), "saved EEF trajectory")
    if not np.array_equal(data["positions"][:, HAND], eef[:, 14:]):
        raise ValueError("Saved hand commands do not match EEF bundle")
    for i in range(n):
        for j, (side, section) in enumerate((("left", slice(0, 7)), ("right", slice(27, 34)))):
            target = model.base_camera[side] @ pose_matrix(eef[i, 7 * j:7 * (j + 1)])
            error = model.error(model.fk(data["positions"][i, section]), target)
            if np.linalg.norm(error[:3]) > 1e-5 or np.linalg.norm(error[3:]) > 1e-4:
                raise ValueError("Saved joints fail EEF FK verification")
    if np.max(abs(np.diff(data["positions"][:, ARM], axis=0))) > report["max_ik_step_rad"]:
        raise ValueError("Saved arm trajectory exceeds the recorded IK step bound")
    playback, _ = trajectory(data["time"], data["positions"], lower, upper, report["minimum_time_scale"],
                             interpolation=report["interpolation"])
    return data, report, playback
