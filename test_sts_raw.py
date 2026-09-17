import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


PORT = "/dev/so101_follower"
STEP = 20

motors = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

bus = FeetechMotorsBus(
    port=PORT,
    motors=motors,
)

bus.connect()

print("\n===== SO-101 RAW MOTOR TEST =====")

try:
    for name, motor in motors.items():

        print("\n" + "=" * 50)
        print(f"{name}   ID={motor.id}")
        print("=" * 50)

        try:
            pos = bus.read(
                "Present_Position",
                name,
                normalize=False
            )

            goal = bus.read(
                "Goal_Position",
                name,
                normalize=False
            )

            torque = bus.read(
                "Torque_Enable",
                name,
                normalize=False
            )

            mode = bus.read(
                "Operating_Mode",
                name,
                normalize=False
            )

            status = bus.read(
                "Status",
                name,
                normalize=False
            )

            print(f"Present_Position : {pos}")
            print(f"Goal_Position    : {goal}")
            print(f"Torque_Enable    : {torque}")
            print(f"Operating_Mode   : {mode}")
            print(f"Status           : {status}")

            # Position mode = 0.
            # Не двигаем мотор, если он почему-то в другом режиме.
            if mode != 0:
                print("❌ SKIP: motor is NOT in position mode")
                continue

            input(
                f"\nPress ENTER to move {name} by ~{STEP} ticks "
                "(Ctrl+C to stop)..."
            )

            # Сначала goal = current position,
            # чтобы при включении torque не было рывка.
            bus.write(
                "Goal_Position",
                name,
                pos,
                normalize=False
            )

            if torque == 0:
                print("Torque was OFF -> enabling temporarily")
                bus.write(
                    "Torque_Enable",
                    name,
                    1,
                    normalize=False
                )
                time.sleep(0.2)

            # Выбираем безопасное направление.
            if pos < 4000 - STEP:
                target = pos + STEP
            else:
                target = pos - STEP

            print(f"COMMAND: {pos} -> {target}")

            bus.write(
                "Goal_Position",
                name,
                target,
                normalize=False
            )

            time.sleep(0.7)

            goal_after = bus.read(
                "Goal_Position",
                name,
                normalize=False
            )

            pos_after = bus.read(
                "Present_Position",
                name,
                normalize=False
            )

            moving = bus.read(
                "Moving",
                name,
                normalize=False
            )

            status_after = bus.read(
                "Status",
                name,
                normalize=False
            )

            print("\nRESULT:")
            print(f"Goal_Position    : {goal_after}")
            print(f"Present_Position : {pos_after}")
            print(f"Delta encoder    : {pos_after - pos}")
            print(f"Moving           : {moving}")
            print(f"Status           : {status_after}")

            # Возвращаем туда, где был.
            bus.write(
                "Goal_Position",
                name,
                pos,
                normalize=False
            )

            time.sleep(0.7)

            returned = bus.read(
                "Present_Position",
                name,
                normalize=False
            )

            print(f"Returned position: {returned}")

        except Exception as e:
            print(f"❌ ERROR: {repr(e)}")

finally:
    bus.disconnect()

print("\n===== TEST FINISHED =====")
