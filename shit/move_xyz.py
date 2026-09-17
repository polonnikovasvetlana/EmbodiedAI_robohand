#!/usr/bin/env python3
"""Small, terminal-driven Cartesian controller for an SO-101 follower.

The inverse and forward kinematics are provided by LeRobot's RobotKinematics
(Placo).  This file deliberately does not implement its own IK.
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
from huggingface_hub import sync_bucket

from lerobot.model import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.robot_utils import precise_sleep


DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_ROBOT_ID = "so101_follower"
EE_FRAME = "gripper_frame_link"

# Motor order must match the order given to RobotKinematics.
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ARM_MOTOR_NAMES = MOTOR_NAMES[:-1]

# Official LeRobot examples pass absolute base-frame XYZ to IK without a frame
# remap. Keep the user frame identical to the SO-101 URDF base frame.
USER_TO_URDF_XYZ = np.array([1.0, 1.0, 1.0], dtype=float)

# Conservative user-frame workspace. Adjust only after checking your setup.
# X: forward, Y: left, Z: physically up, all in metres.
WORKSPACE_MIN = np.array([0.05, -0.30, -0.05], dtype=float)
WORKSPACE_MAX = np.array([0.42, 0.30, 0.40], dtype=float)
MIN_DISTANCE_FROM_BASE_M = 0.07
MAX_DISTANCE_FROM_BASE_M = 0.48

# Motion safety. Each command is interpolated at CONTROL_HZ. LeRobot also
# clamps every individual motor command to MAX_RELATIVE_TARGET from feedback.
CONTROL_HZ = 20.0
MAX_ARM_SPEED_DEG_S = 20.0
MAX_GRIPPER_SPEED_UNITS_S = 25.0
MAX_RELATIVE_TARGET = 2.0
MAX_CARTESIAN_WAYPOINT_M = 0.005

# Gripper commands use its calibrated 0..100 scale (not degrees): 100 is fully
# open, 0 is fully closed. Closing stops on a stable position before zero,
# which indicates that an object or another mechanical obstruction was met.
GRIPPER_UPDATE_PERIOD_S = 0.10
GRIPPER_COMMAND_AHEAD = 4.0
GRIPPER_HOLD_OFFSET = 1.5
GRIPPER_ENDPOINT_TOLERANCE = 1.5
GRIPPER_STALL_WINDOW = 8
GRIPPER_STALL_MOVEMENT = 0.5
GRIPPER_TIMEOUT_S = 10.0

# The SO-101 has only five arm joints and cannot realize an arbitrary 6-DoF
# pose. LeRobot explicitly supports 0.0 for position-only IK. This makes XYZ of
# gripper_frame_link the priority and lets orientation change as needed.
IK_ORIENTATION_WEIGHT = 0.0
IK_MAX_ITERATIONS = 150
IK_POSITION_TOLERANCE_M = 0.008

GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 0.0


def ensure_so101_urdf() -> Path:
    """Fetch the official SO-101 URDF bundle exactly as LeRobot's example does."""
    dest_dir = HF_LEROBOT_HOME / "robot-urdfs" / "so101"
    urdf_path = dest_dir / "so101_new_calib.urdf"
    marker = dest_dir / ".sync_complete"

    if not marker.exists() or not urdf_path.is_file():
        dest_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading the official SO-101 URDF bundle from Hugging Face...")
        try:
            sync_bucket(
                "hf://buckets/lerobot/robot-urdfs/so101",
                str(dest_dir),
                quiet=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not download the official SO-101 URDF bundle. "
                "Check the network, or run with --urdf /path/to/so101_new_calib.urdf."
            ) from exc
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF was not found after download: {urdf_path}")
        marker.touch()

    return urdf_path


def load_joint_limits_deg(urdf_path: Path) -> dict[str, tuple[float, float]]:
    """Read the official URDF limits; this is validation, not an IK solver."""
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        limit = joint.find("limit")
        if name in ARM_MOTOR_NAMES and limit is not None:
            low = math.degrees(float(limit.attrib["lower"]))
            high = math.degrees(float(limit.attrib["upper"]))
            limits[name] = (low, high)

    missing = set(ARM_MOTOR_NAMES) - set(limits)
    if missing:
        raise ValueError(f"URDF has no limits for: {sorted(missing)}")
    return limits


def read_joints(robot: SO101Follower) -> np.ndarray:
    observation = robot.get_observation()
    return np.array([float(observation[f"{name}.pos"]) for name in MOTOR_NAMES], dtype=float)


def joint_action(joints: np.ndarray) -> dict[str, float]:
    return {f"{name}.pos": float(joints[i]) for i, name in enumerate(MOTOR_NAMES)}


def format_joints(joints: np.ndarray) -> str:
    values = []
    for i, name in enumerate(MOTOR_NAMES):
        unit = "%" if name == "gripper" else "deg"
        values.append(f"{name}={joints[i]:.2f} {unit}")
    return ", ".join(values)


def user_to_urdf_xyz(xyz: np.ndarray) -> np.ndarray:
    """Convert user XYZ to the official URDF base frame (currently identity)."""
    return np.asarray(xyz, dtype=float) * USER_TO_URDF_XYZ


def urdf_to_user_xyz(xyz: np.ndarray) -> np.ndarray:
    """Convert official FK output to user XYZ (currently identity)."""
    return np.asarray(xyz, dtype=float) * USER_TO_URDF_XYZ


def print_state(robot: SO101Follower, kinematics: RobotKinematics) -> np.ndarray:
    joints = read_joints(robot)
    pose = kinematics.forward_kinematics(joints)
    xyz = urdf_to_user_xyz(pose[:3, 3])
    print(f"Current joints: {format_joints(joints)}")
    print(
        f"Current EE XYZ: x={xyz[0]:.4f}, y={xyz[1]:.4f}, z={xyz[2]:.4f} m "
        f"= ({xyz[0] * 1000:.1f}, {xyz[1] * 1000:.1f}, {xyz[2] * 1000:.1f}) mm"
    )
    return joints


def print_axes() -> None:
    print("Official SO-101 base frame (origin = URDF base_link):")
    print("  +X: forward from the base")
    print("  +Y: left when looking in the +X direction")
    print("  +Z: upward")
    print("  -Z: downward")
    print("  Relative commands x/y/z use millimetres.")


def validate_xyz(xyz: np.ndarray) -> None:
    if not np.all(np.isfinite(xyz)):
        raise ValueError("XYZ values must be finite numbers.")
    if np.any(xyz < WORKSPACE_MIN) or np.any(xyz > WORKSPACE_MAX):
        raise ValueError(
            "Target is outside the configured workspace: "
            f"x=[{WORKSPACE_MIN[0]:.2f}, {WORKSPACE_MAX[0]:.2f}], "
            f"y=[{WORKSPACE_MIN[1]:.2f}, {WORKSPACE_MAX[1]:.2f}], "
            f"z=[{WORKSPACE_MIN[2]:.2f}, {WORKSPACE_MAX[2]:.2f}] m."
        )
    distance = float(np.linalg.norm(xyz))
    if not MIN_DISTANCE_FROM_BASE_M <= distance <= MAX_DISTANCE_FROM_BASE_M:
        raise ValueError(
            f"Target distance from the base is {distance:.3f} m; allowed range is "
            f"[{MIN_DISTANCE_FROM_BASE_M:.2f}, {MAX_DISTANCE_FROM_BASE_M:.2f}] m."
        )


def rotation_error_deg(desired: np.ndarray, actual: np.ndarray) -> float:
    relative = desired.T @ actual
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def solve_ik(
    kinematics: RobotKinematics,
    current_joints: np.ndarray,
    requested_xyz: np.ndarray,
    fixed_rotation: np.ndarray,
    joint_limits: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, float, float]:
    desired_pose = np.eye(4, dtype=float)
    desired_pose[:3, :3] = fixed_rotation
    desired_pose[:3, 3] = requested_xyz

    q = current_joints.copy()
    for _ in range(IK_MAX_ITERATIONS):
        previous = q.copy()
        q = np.asarray(
            kinematics.inverse_kinematics(
                q,
                desired_pose,
                orientation_weight=IK_ORIENTATION_WEIGHT,
            ),
            dtype=float,
        )
        # The gripper joint does not affect gripper_frame_link. Preserve its
        # measured opening explicitly rather than trusting an unconstrained IK value.
        q[-1] = current_joints[-1]
        if not np.all(np.isfinite(q)):
            raise RuntimeError("IK returned NaN/Inf joint values.")
        if float(np.max(np.abs(q - previous))) < 1e-6:
            break

    reached_pose = kinematics.forward_kinematics(q)
    position_error = float(np.linalg.norm(reached_pose[:3, 3] - requested_xyz))
    orientation_error = rotation_error_deg(fixed_rotation, reached_pose[:3, :3])

    if position_error > IK_POSITION_TOLERANCE_M:
        raise RuntimeError(
            f"IK did not converge closely enough (XYZ error {position_error * 1000:.1f} mm, "
            f"limit {IK_POSITION_TOLERANCE_M * 1000:.1f} mm). Nothing was moved."
        )

    violations = []
    for i, name in enumerate(ARM_MOTOR_NAMES):
        low, high = joint_limits[name]
        if q[i] < low - 1e-3 or q[i] > high + 1e-3:
            violations.append(f"{name}={q[i]:.2f} deg not in [{low:.2f}, {high:.2f}]")
    if violations:
        raise RuntimeError("IK solution exceeds URDF joint limits: " + "; ".join(violations))

    return q, position_error, orientation_error


def move_slowly(robot: SO101Follower, target: np.ndarray) -> None:
    """Interpolate in joint space and send small, speed-limited commands."""
    start = read_joints(robot)
    delta = target - start
    arm_duration = float(np.max(np.abs(delta[:-1]))) / MAX_ARM_SPEED_DEG_S
    gripper_duration = abs(float(delta[-1])) / MAX_GRIPPER_SPEED_UNITS_S
    duration = max(arm_duration, gripper_duration)
    steps = max(1, math.ceil(duration * CONTROL_HZ))

    started_at = time.perf_counter()
    for step in range(1, steps + 1):
        alpha = step / steps
        command = start + alpha * delta
        robot.send_action(joint_action(command))
        next_tick = started_at + step / CONTROL_HZ
        precise_sleep(max(0.0, next_tick - time.perf_counter()))


def read_gripper(robot: SO101Follower) -> float:
    return float(robot.get_observation()["gripper.pos"])


def send_gripper(robot: SO101Follower, position: float) -> None:
    # Send only the gripper key. Re-sending measured arm joints can slowly
    # ratchet a gravity-loaded arm downwards.
    robot.send_action({"gripper.pos": float(np.clip(position, 0.0, 100.0))})


def open_gripper_fully(robot: SO101Follower) -> float:
    """Open to calibrated 100% and confirm the endpoint from feedback."""
    deadline = time.monotonic() + GRIPPER_TIMEOUT_S
    positions: deque[float] = deque(maxlen=GRIPPER_STALL_WINDOW)

    while time.monotonic() < deadline:
        current = read_gripper(robot)
        if current >= GRIPPER_OPEN - GRIPPER_ENDPOINT_TOLERANCE:
            send_gripper(robot, GRIPPER_OPEN)
            print(f"Gripper fully open: {current:.1f}% (goal 100%).")
            return current

        positions.append(current)
        send_gripper(robot, min(GRIPPER_OPEN, current + GRIPPER_COMMAND_AHEAD))
        precise_sleep(GRIPPER_UPDATE_PERIOD_S)

        if (
            len(positions) == positions.maxlen
            and max(positions) - min(positions) < GRIPPER_STALL_MOVEMENT
        ):
            send_gripper(robot, current)
            raise RuntimeError(
                f"Gripper stopped at {current:.1f}% before fully opening. "
                "Check for an obstruction or recalibrate it."
            )

    current = read_gripper(robot)
    send_gripper(robot, current)
    raise RuntimeError(f"Gripper did not fully open within {GRIPPER_TIMEOUT_S:.0f} seconds.")


def close_gripper_until_contact(robot: SO101Follower) -> tuple[float, bool]:
    """Close gradually; stop and hold when position feedback indicates contact."""
    deadline = time.monotonic() + GRIPPER_TIMEOUT_S
    positions: deque[float] = deque(maxlen=GRIPPER_STALL_WINDOW)

    while time.monotonic() < deadline:
        current = read_gripper(robot)
        if current <= GRIPPER_CLOSED + GRIPPER_ENDPOINT_TOLERANCE:
            send_gripper(robot, GRIPPER_CLOSED)
            print(f"Gripper fully closed: {current:.1f}%. No object was detected.")
            return current, False

        positions.append(current)
        send_gripper(robot, max(GRIPPER_CLOSED, current - GRIPPER_COMMAND_AHEAD))
        precise_sleep(GRIPPER_UPDATE_PERIOD_S)

        if (
            len(positions) == positions.maxlen
            and max(positions) - min(positions) < GRIPPER_STALL_MOVEMENT
        ):
            current = read_gripper(robot)
            # A small goal offset keeps gentle gripping force after contact.
            hold_position = max(GRIPPER_CLOSED, current - GRIPPER_HOLD_OFFSET)
            send_gripper(robot, hold_position)
            print(
                f"Contact/object detected at {current:.1f}%. "
                f"Holding with goal {hold_position:.1f}%."
            )
            return current, True

    current = read_gripper(robot)
    send_gripper(robot, current)
    raise RuntimeError(f"Gripper close timed out at {current:.1f}%.")


def move_to_xyz(
    robot: SO101Follower,
    kinematics: RobotKinematics,
    requested_xyz: np.ndarray,
    fixed_rotation: np.ndarray,
    joint_limits: dict[str, tuple[float, float]],
    current_joints: np.ndarray | None = None,
) -> None:
    # requested_xyz is in the same absolute base frame used by the official
    # LeRobot examples and RobotKinematics.
    print(
        f"Requested XYZ: x={requested_xyz[0]:.4f}, "
        f"y={requested_xyz[1]:.4f}, z={requested_xyz[2]:.4f} m "
        f"({requested_xyz[0] * 1000:.1f}, {requested_xyz[1] * 1000:.1f}, "
        f"{requested_xyz[2] * 1000:.1f} mm)"
    )
    validate_xyz(requested_xyz)
    requested_urdf_xyz = user_to_urdf_xyz(requested_xyz)

    if current_joints is None:
        current_joints = print_state(robot, kinematics)

    # Plan the whole straight Cartesian path before moving anything. Directly
    # interpolating only the final joint angles can make the shoulder/wrist
    # reconfigure while the gripper tip barely moves. Sequential IK waypoints
    # keep gripper_frame_link on the requested line.
    start_urdf_xyz = kinematics.forward_kinematics(current_joints)[:3, 3].copy()
    distance_m = float(np.linalg.norm(requested_urdf_xyz - start_urdf_xyz))
    waypoint_count = max(1, math.ceil(distance_m / MAX_CARTESIAN_WAYPOINT_M))
    path: list[np.ndarray] = []
    q_guess = current_joints.copy()
    position_error = 0.0
    orientation_error = 0.0

    for step in range(1, waypoint_count + 1):
        alpha = step / waypoint_count
        waypoint_xyz = start_urdf_xyz + alpha * (requested_urdf_xyz - start_urdf_xyz)
        q_guess, position_error, orientation_error = solve_ik(
            kinematics,
            q_guess,
            waypoint_xyz,
            fixed_rotation,
            joint_limits,
        )
        path.append(q_guess.copy())

    target_joints = path[-1]
    print(f"IK joint target: {format_joints(target_joints)}")
    print(f"IK check: XYZ error={position_error * 1000:.1f} mm (position-only IK)")
    print(f"Moving gripper tip along {waypoint_count} Cartesian waypoint(s)...")
    for waypoint_joints in path:
        move_slowly(robot, waypoint_joints)
    print_state(robot, kinematics)


def run_terminal(robot: SO101Follower, kinematics: RobotKinematics, urdf_path: Path) -> None:
    joint_limits = load_joint_limits_deg(urdf_path)
    startup_joints = print_state(robot, kinematics)
    startup_pose = kinematics.forward_kinematics(startup_joints)
    fixed_rotation = startup_pose[:3, :3].copy()
    home_joints = startup_joints.copy()

    print("\nThe current pose is saved as 'home'.")
    print("IK controls gripper position; orientation is free (position-only mode).")
    print_axes()
    print("Commands:")
    print("  123 0 20 - absolute XYZ in millimetres")
    print("  xyz 0.123 0 0.020 - absolute XYZ in metres")
    print("  x 15 / y -10 / z 5 - relative motion on one axis in millimetres")
    print("  open | close | home | status | axes | help | quit")

    while True:
        try:
            command = input("so101> ").strip()
        except EOFError:
            command = "quit"

        if not command:
            continue
        lowered = command.lower()

        if lowered in {"quit", "exit", "q"}:
            return

        try:
            if lowered == "open":
                print("Opening gripper...")
                open_gripper_fully(robot)
                print_state(robot, kinematics)
                continue

            if lowered == "close":
                print("Closing gripper until contact/object or the closed endpoint...")
                close_gripper_until_contact(robot)
                print_state(robot, kinematics)
                continue

            if lowered == "home":
                target = home_joints.copy()
                # Home changes only the arm; retain the current gripper opening.
                target[-1] = read_joints(robot)[-1]
                print("Moving to the startup home pose...")
                move_slowly(robot, target)
                print_state(robot, kinematics)
                continue

            if lowered == "status":
                print_state(robot, kinematics)
                continue

            if lowered == "axes":
                print_axes()
                print_state(robot, kinematics)
                continue

            if lowered == "help":
                print("123 0 20 = absolute millimetres")
                print("xyz 0.123 0 0.020 = absolute metres")
                print("x 15, y -10, z 5 = relative millimetres on only that axis")
                print("open | close | home | status | axes | quit")
                continue

            fields = command.replace(",", " ").split()
            first = fields[0].lower()

            if first in {"x", "y", "z"}:
                if len(fields) != 2:
                    raise ValueError("Axis command format: x 15, y -10, or z 5 (millimetres).")
                delta_mm = float(fields[1])
                current_joints = print_state(robot, kinematics)
                requested_xyz = urdf_to_user_xyz(
                    kinematics.forward_kinematics(current_joints)[:3, 3]
                )
                axis = {"x": 0, "y": 1, "z": 2}[first]
                requested_xyz[axis] += delta_mm / 1000.0
                print(f"Relative {first.upper()} motion: {delta_mm:+.1f} mm")
                move_to_xyz(
                    robot,
                    kinematics,
                    requested_xyz,
                    fixed_rotation,
                    joint_limits,
                    current_joints,
                )
                continue

            if first == "xyz":
                if len(fields) != 4:
                    raise ValueError("Metre command format: xyz 0.123 0 0.020")
                requested_xyz = np.array([float(value) for value in fields[1:]], dtype=float)
            else:
                if len(fields) != 3:
                    raise ValueError(
                        "Use: 123 0 20 (mm), xyz 0.123 0 0.020 (m), "
                        "x 15/y -10/z 5 (mm), or open/close/home/quit."
                    )
                requested_xyz = np.array([float(value) for value in fields], dtype=float) / 1000.0

            move_to_xyz(
                robot,
                kinematics,
                requested_xyz,
                fixed_rotation,
                joint_limits,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DEFAULT_PORT, help="SO-101 serial port")
    parser.add_argument("--id", default=DEFAULT_ROBOT_ID, help="LeRobot calibration id")
    parser.add_argument(
        "--urdf",
        type=Path,
        help="Optional local so101_new_calib.urdf; otherwise use the official HF bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urdf_path = args.urdf.expanduser().resolve() if args.urdf else ensure_so101_urdf()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        use_degrees=True,
        max_relative_target=MAX_RELATIVE_TARGET,
    )
    robot = SO101Follower(config)
    if not robot.calibration_fpath.is_file() or not robot.calibration:
        raise FileNotFoundError(
            f"Saved calibration for id '{args.id}' was not found at "
            f"{robot.calibration_fpath}. Run lerobot-calibrate first."
        )

    print(f"LeRobot calibration: {robot.calibration_fpath}")
    print(f"Official SO-101 URDF: {urdf_path}")
    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name=EE_FRAME,
        joint_names=MOTOR_NAMES,
    )

    try:
        robot.connect()
        if not robot.is_connected:
            raise RuntimeError(f"Robot did not connect on {args.port}")
        print(f"Connected to {args.id} on {args.port}")
        run_terminal(robot, kinematics, urdf_path)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if robot.is_connected:
            robot.disconnect()
            print("Robot disconnected; torque disabled.")


if __name__ == "__main__":
    main()
