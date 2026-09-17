from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

motors = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

bus = FeetechMotorsBus(
    port="/dev/so101_follower",
    motors=motors,
)

bus.connect()

for name in motors:
    print(f"\n{name}  ID={motors[name].id}")

    for reg in [
        "Present_Position",
        "Goal_Position",
        "Min_Position_Limit",
        "Max_Position_Limit",
        "Homing_Offset",
        "Operating_Mode",
        "Torque_Enable",
        "Status",
    ]:
        try:
            value = bus.read(reg, name, normalize=False)
            print(f"{reg:22}: {value}")
        except Exception as e:
            print(f"{reg:22}: ERROR {e}")

bus.disconnect()
