import time
import numpy as np

from scipy.spatial.transform import Rotation

from lerobot.lerobot_types import RobotAction
from lerobot.model.kinematics import RobotKinematics

from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)

from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig,
)

from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)


# ============================================================
# SETTINGS
# ============================================================

PORT = "/dev/ttyACM0"
ROBOT_ID = "iw_so101"

URDF_PATH = (
    "/home/svetlana/"
    "SO-ARM100/Simulation/SO101/"
    "so101_new_calib.urdf"
)

FPS = 30

# Cartesian workspace in meters.
# Это safety box, а не реальный физический workspace.
EE_BOUNDS = {
    "min": [-0.35, -0.35, 0.03],
    "max": [0.35, 0.35, 0.40],
}

# Максимальный Cartesian шаг за один control frame
MAX_EE_STEP_M = 0.01


# ============================================================
# ROBOT
# ============================================================

robot_config = SO101FollowerConfig(
    port=PORT,
    id=ROBOT_ID,
    use_degrees=True,
    max_relative_target=15.0,
)

robot = SO101Follower(robot_config)


# ============================================================
# CONNECT
# ============================================================

print("Connecting SO-101...")

robot.connect()

print("Connected.")


# ============================================================
# KINEMATICS
# ============================================================

motor_names = list(
    robot.bus.motors.keys()
)

print()
print("Motor order:")
print(motor_names)
print()

kinematics = RobotKinematics(
    urdf_path=URDF_PATH,
    target_frame_name="gripper_frame_link",
    joint_names=motor_names,
)


# ============================================================
# OFFICIAL LEROBOT EE -> JOINT PIPELINE
# ============================================================

ee_to_joints = RobotProcessorPipeline[
    tuple[RobotAction, dict],
    RobotAction,
](
    steps=[
        EEBoundsAndSafety(
            end_effector_bounds=EE_BOUNDS,
            max_ee_step_m=MAX_EE_STEP_M,
            raise_on_jump=False,
        ),

        InverseKinematicsEEToJoints(
            kinematics=kinematics,
            motor_names=motor_names,
            initial_guess_current_joints=True,
            orientation_weight=0.0,
        ),
    ],

    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)


# ============================================================
# CURRENT POSE
# ============================================================

def get_current_pose():

    obs = robot.get_observation()

    q = np.array(
        [
            float(obs[f"{name}.pos"])
            for name in motor_names
        ],
        dtype=float,
    )

    T = kinematics.forward_kinematics(
        q
    )

    xyz = T[:3, 3]

    rotvec = Rotation.from_matrix(
        T[:3, :3]
    ).as_rotvec()

    gripper = float(
        obs["gripper.pos"]
    )

    return (
        obs,
        xyz,
        rotvec,
        gripper,
    )


# ============================================================
# PRINT CURRENT POSITION
# ============================================================

def print_position():

    obs, xyz, rotvec, gripper = (
        get_current_pose()
    )

    print()
    print("Current XYZ:")

    print(
        f"X = {xyz[0] * 1000:.1f} mm"
    )

    print(
        f"Y = {xyz[1] * 1000:.1f} mm"
    )

    print(
        f"Z = {xyz[2] * 1000:.1f} mm"
    )

    print()
    print(
        f"Gripper = {gripper:.1f}"
    )

    print()


# ============================================================
# GRIPPER
# ============================================================

def set_gripper(value):

    value = float(
        np.clip(
            value,
            0.0,
            100.0,
        )
    )

    obs = robot.get_observation()

    action = {}

    # Держим все остальные суставы
    # в текущем положении.
    for name in motor_names:

        action[f"{name}.pos"] = float(
            obs[f"{name}.pos"]
        )

    action["gripper.pos"] = value

    robot.send_action(
        action
    )

    print()
    print(
        f"Gripper -> {value:.1f}"
    )
    print()


# ============================================================
# MOVE XYZ
# ============================================================

def move_xyz(
    x_mm,
    y_mm,
    z_mm,
):

    # mm -> m
    target = np.array(
        [
            x_mm / 1000.0,
            y_mm / 1000.0,
            z_mm / 1000.0,
        ],
        dtype=float,
    )

    (
        obs,
        current_xyz,
        rotvec,
        gripper,
    ) = get_current_pose()


    # ========================================================
    # CLAMP TARGET TO SAFETY BOX
    # ========================================================

    original_target = target.copy()

    target = np.clip(
        target,
        EE_BOUNDS["min"],
        EE_BOUNDS["max"],
    )

    if not np.allclose(
        original_target,
        target,
    ):

        print()
        print(
            "Target was outside safety box."
        )

        print(
            "Clamped target:"
        )

        print(
            f"X={target[0] * 1000:.1f} "
            f"Y={target[1] * 1000:.1f} "
            f"Z={target[2] * 1000:.1f} mm"
        )


    # ========================================================
    # DISTANCE
    # ========================================================

    distance = np.linalg.norm(
        target
        - current_xyz
    )


    # Скорость примерно 40 mm/s
    duration = max(
        1.0,
        distance / 0.04,
    )

    steps = max(
        1,
        int(
            duration * FPS
        ),
    )


    print()
    print(
        "Current XYZ:"
    )

    print(
        f"{current_xyz[0] * 1000:.1f} "
        f"{current_xyz[1] * 1000:.1f} "
        f"{current_xyz[2] * 1000:.1f} mm"
    )

    print()

    print(
        "Target XYZ:"
    )

    print(
        f"{target[0] * 1000:.1f} "
        f"{target[1] * 1000:.1f} "
        f"{target[2] * 1000:.1f} mm"
    )

    print()

    print(
        f"Distance: "
        f"{distance * 1000:.1f} mm"
    )

    print(
        f"Move time: "
        f"~{duration:.1f} s"
    )


    # Reset внутреннего состояния processor pipeline
    ee_to_joints.reset()


    # ========================================================
    # SMOOTH CARTESIAN MOVEMENT
    # ========================================================

    for i in range(
        1,
        steps + 1,
    ):

        alpha = i / steps

        # smoothstep
        alpha = (
            alpha
            * alpha
            * (
                3.0
                - 2.0 * alpha
            )
        )

        xyz = (
            current_xyz
            + alpha
            * (
                target
                - current_xyz
            )
        )


        # ====================================================
        # CARTESIAN ACTION
        # ====================================================

        ee_action = {
            "ee.x": float(
                xyz[0]
            ),

            "ee.y": float(
                xyz[1]
            ),

            "ee.z": float(
                xyz[2]
            ),

            # Пока сохраняем текущую orientation
            "ee.wx": float(
                rotvec[0]
            ),

            "ee.wy": float(
                rotvec[1]
            ),

            "ee.wz": float(
                rotvec[2]
            ),

            # Gripper остаётся в текущем положении
            "ee.gripper_pos": float(
                gripper
            ),
        }


        # Актуальное состояние суставов
        obs = robot.get_observation()


        # ====================================================
        # OFFICIAL LEROBOT PROCESSOR
        #
        # EE xyz
        # ↓
        # safety
        # ↓
        # IK
        # ↓
        # joint commands
        # ====================================================

        joint_action = ee_to_joints(
            (
                ee_action,
                obs,
            )
        )


        robot.send_action(
            joint_action
        )


        time.sleep(
            1.0 / FPS
        )


    print()
    print(
        "Target command finished."
    )

    print_position()


# ============================================================
# HELP
# ============================================================

def print_help():

    print()
    print(
        "========================================"
    )

    print(
        "SO-101 Cartesian Controller"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Move to XYZ [mm]:"
    )

    print(
        "    250 0 120"
    )

    print()

    print(
        "Gripper:"
    )

    print(
        "    open"
    )

    print(
        "    close"
    )

    print(
        "    grip 50"
    )

    print()

    print(
        "Other:"
    )

    print(
        "    p      current XYZ"
    )

    print(
        "    help   commands"
    )

    print(
        "    q      quit"
    )

    print()

    print(
        "========================================"
    )

    print()


# ============================================================
# START INFO
# ============================================================

print_help()

print_position()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        text = input(
            "SO101 > "
        ).strip()

        if not text:
            continue


        # ====================================================
        # QUIT
        # ====================================================

        if text.lower() in [
            "q",
            "quit",
            "exit",
        ]:

            break


        # ====================================================
        # POSITION
        # ====================================================

        if text.lower() == "p":

            print_position()

            continue


        # ====================================================
        # HELP
        # ====================================================

        if text.lower() in [
            "help",
            "h",
        ]:

            print_help()

            continue


        # ====================================================
        # OPEN
        # ====================================================

        if text.lower() == "open":

            set_gripper(
                100.0
            )

            continue


        # ====================================================
        # CLOSE
        # ====================================================

        if text.lower() == "close":

            set_gripper(
                0.0
            )

            continue


        # ====================================================
        # CUSTOM GRIPPER
        #
        # grip 50
        # ====================================================

        if text.lower().startswith(
            "grip "
        ):

            try:

                parts = text.split()

                if len(parts) != 2:

                    raise ValueError()

                value = float(
                    parts[1]
                )

                set_gripper(
                    value
                )

            except Exception:

                print()
                print(
                    "Use:"
                )

                print(
                    "grip 0..100"
                )

                print()

            continue


        # ====================================================
        # XYZ
        # ====================================================

        try:

            values = (
                text.split()
            )

            if len(values) != 3:

                print()
                print(
                    "Unknown command."
                )

                print(
                    "Type 'help'"
                )

                print()

                continue


            x, y, z = map(
                float,
                values,
            )

            move_xyz(
                x,
                y,
                z,
            )


        except Exception as e:

            print()
            print(
                "MOVE ERROR:"
            )

            print(e)

            print()


# ============================================================
# CTRL+C
# ============================================================

except KeyboardInterrupt:

    print()
    print(
        "Stopped by CTRL+C"
    )


# ============================================================
# DISCONNECT
# ============================================================

finally:

    print()
    print(
        "Disconnecting..."
    )

    robot.disconnect()

    print(
        "Robot disconnected."
    )