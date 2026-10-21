from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (ARM, HAND, Kinematics, decode, joint_names, load_bundle, pose_matrix,
                  read_episode, read_limits, save_bundle, trajectory)
from live import stream

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Convert_data/outputs/fr3_wuji/tomato_eef_cam0"


@pytest.fixture
def real_episode():
    if not DATA.exists():
        pytest.skip("Local example dataset is not installed")
    calibration = yaml.safe_load((DATA / "conversion/calibration.yaml").read_text())
    eef, seeds, times, frames, path = read_episode(DATA, 0, end=40)
    lower, upper = read_limits(ROOT / "limits.yaml")
    return calibration, eef, seeds, times, lower, upper


def test_both_arm_frames_and_hands_roundtrip(real_episode):
    calibration, eef, seeds, times, lower, upper = real_episode
    model = Kinematics(calibration)
    q, targets, errors = decode(eef, seeds, model, lower, upper)
    assert np.max(errors[:, :, 0]) < 1e-6
    assert np.max(errors[:, :, 1]) < 1e-6
    np.testing.assert_array_equal(q[:, HAND], eef[:, 14:])
    np.testing.assert_array_equal(q[:, ARM], seeds)
    for j, side in enumerate(("left", "right")):
        np.testing.assert_allclose(targets[0, j], model.base_camera[side] @ pose_matrix(eef[0, j*7:(j+1)*7]))


def test_ik_uses_eef_target_instead_of_blindly_copying_seeds(real_episode):
    calibration, eef, seeds, times, lower, upper = real_episode
    model = Kinematics(calibration)
    changed = seeds[0, :7].copy()
    changed[0] += 0.015
    target = model.fk(changed)
    result, error = model.ik(target, seeds[0, :7], lower[:7], upper[:7])
    assert np.linalg.norm(result - seeds[0, :7]) > 0.005
    assert np.linalg.norm(error[:3]) < 1e-5
    assert np.linalg.norm(error[3:]) < 1e-4
    target[:3, 3] = [10, 10, 10]
    with pytest.raises(ValueError, match="IK failed"):
        model.ik(target, seeds[0, :7], lower[:7], upper[:7])


def test_quaternion_order_sign_and_rejection():
    pose = [0.1, 0.2, 0.3, *Rotation.from_euler("z", np.pi / 2).as_quat()]
    t = pose_matrix(pose)
    np.testing.assert_allclose(t[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    pose[3:] = -np.asarray(pose[3:])
    np.testing.assert_allclose(pose_matrix(pose), t)
    with pytest.raises(ValueError, match="Quaternion"):
        pose_matrix([0] * 7)


def test_branch_jump_and_limit_rejection(real_episode):
    calibration, eef, seeds, times, lower, upper = real_episode
    model = Kinematics(calibration)
    bad = eef.copy()
    bad[0, 14] = 100
    with pytest.raises(ValueError, match="Joint limit"):
        decode(bad, seeds, model, lower, upper)
    with pytest.raises(ValueError, match="branch jump"):
        decode(eef, seeds, model, lower, upper, max_step=1e-8)


def test_bounded_smoothing_stays_inside_limits_and_derivative_caps():
    times = np.arange(8) / 30
    q = np.zeros((8, 54))
    q[:, 0] = [0, .8, 1, 1, .2, -.9, -1, 0]
    curve, report = trajectory(times, q, np.full(54, -1), np.ones(54), minimum_scale=2.5)
    t = np.linspace(0, curve.duration, 10001)
    samples = curve.sample(t)
    assert samples.min() >= -1 - 1e-12 and samples.max() <= 1 + 1e-12
    assert report["time_scale"] >= 2.5
    assert abs(curve.sample(t, 1)).max() <= .42 + 1e-8
    assert abs(curve.sample(t, 2)).max() <= 1.44 + 1e-8
    assert abs(curve.sample(t, 3)).max() <= 17.28 + 1e-8
    for end in (0, curve.duration):
        np.testing.assert_allclose(curve.sample(end, 1), 0, atol=1e-10)
        np.testing.assert_allclose(curve.sample(end, 2), 0, atol=1e-10)
    with pytest.raises(ValueError, match="timestamps"):
        trajectory(np.zeros(8), q, np.full(54, -1), np.ones(54))


def test_bundle_hash_and_fk_revalidation(real_episode, tmp_path):
    calibration, eef, seeds, times, lower, upper = real_episode
    q, _, _ = decode(eef, seeds, Kinematics(calibration), lower, upper)
    report = {"schema": "fr3-wuji-eef-replay-v1", "joint_names": joint_names(),
              "minimum_time_scale": 2.5, "interpolation": "bounded", "calibration": calibration,
              "max_ik_step_rad": .3}
    arrays = {"positions": q, "time": times, "lower": lower, "upper": upper, "eef": eef}
    output = tmp_path / "bundle"
    save_bundle(output, arrays, report)
    loaded, _, _ = load_bundle(output)
    np.testing.assert_array_equal(loaded["positions"], q)
    with (output / "trajectory.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_bundle(output)


class FakeClock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeDevices:
    def __init__(self, clock, late=False):
        self.clock, self.late = clock, late
        self.frames = []
        self.stopped = self.closed = self.finished = False

    def prepare(self, player, feedback, now):
        self.start = player.plan.start

    def submit(self, frame, now):
        assert frame.sequence == len(self.frames)
        assert now < frame.valid_until
        if not self.frames:
            np.testing.assert_array_equal(frame.positions, self.start)
        self.frames.append(frame)
        if self.late:
            self.clock.now += .05
        return SimpleNamespace(positions=frame.positions, velocities=np.zeros(54), epoch=0,
            hand_control="strict", hand_targets_reached=True,
            check=lambda *args: None, hands_settled=lambda *args: True)

    def finish(self):
        assert self.frames[-1].phase == "settle_end"
        self.finished = True
        return {"state": "released", "physical_stop_confirmed": False}

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_stream_finishes_and_missed_deadline_stops_without_catchup():
    q = np.zeros((2, 54))
    curve, _ = trajectory([0., .1], q, np.full(54, -1), np.ones(54))
    for late in (False, True):
        clock = FakeClock()
        devices = FakeDevices(clock, late)
        def run():
            return stream(devices, curve, curve, q[0], "test", SimpleNamespace,
                stop_requested=lambda: False, clock=clock, sleep=clock.sleep, settle_timeout=2)
        if late:
            with pytest.raises(RuntimeError, match="deadline missed"):
                run()
            assert devices.stopped and devices.closed and len(devices.frames) == 1
        else:
            result, history = run()
            assert devices.finished and result["state"] == "released"
            assert len(history["command"]) > 50


def test_operator_stop_requests_device_stop():
    clock = FakeClock()
    devices = FakeDevices(clock)
    curve, _ = trajectory([0., 1.], np.zeros((2, 54)), np.full(54, -1), np.ones(54))
    with pytest.raises(InterruptedError):
        stream(devices, curve, curve, np.zeros(54), "test", SimpleNamespace,
               stop_requested=lambda: True, clock=clock, sleep=clock.sleep)
    assert devices.stopped and devices.closed and not devices.frames
