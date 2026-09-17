from so101_control import SO101Arm


def main():

    arm = SO101Arm()

    print()
    print("SO-101 READY")
    print()
    print("Commands:")
    print("  pose")
    print("  move X Y Z")
    print("  rel DX DY DZ")
    print("  open")
    print("  close")
    print("  gripper POSITION")
    print("  current")
    print("  lock")
    print("  free")
    print("  stop")
    print("  quit")
    print()

    try:

        while True:

            command = input(
                "SO101 > "
            ).strip()

            if not command:
                continue

            parts = command.split()

            name = parts[0].lower()

            # ================================================
            # POSE
            # ================================================

            if name == "pose":

                pose = arm.get_pose()

                print(
                    f"x={pose.x:.1f} mm  "
                    f"y={pose.y:.1f} mm  "
                    f"z={pose.z:.1f} mm"
                )

            # ================================================
            # MOVE
            # ================================================

            elif name == "move":

                if len(parts) != 4:

                    print(
                        "Usage: move X Y Z"
                    )

                    continue

                result = arm.move(
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )

                print(result)

            # ================================================
            # RELATIVE MOVE
            # ================================================

            elif name == "rel":

                if len(parts) != 4:

                    print(
                        "Usage: rel DX DY DZ"
                    )

                    continue

                result = arm.move_relative(
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )

                print(result)

            # ================================================
            # OPEN
            # ================================================

            elif name == "open":

                print(
                    arm.open()
                )

            # ================================================
            # CLOSE
            # ================================================

            elif name == "close":

                print(
                    arm.close()
                )

            # ================================================
            # RAW GRIPPER
            # ================================================

            elif name == "gripper":

                if len(parts) != 2:

                    print(
                        "Usage: gripper POSITION"
                    )

                    continue

                print(
                    arm.set_gripper(
                        float(parts[1])
                    )
                )

            # ================================================
            # GRIPPER CURRENT
            # ================================================

            elif name == "current":

                print(
                    f"Gripper current: "
                    f"{arm.get_gripper_current():.3f} A"
                )

            # ================================================
            # LOCK CURRENT ORIENTATION
            # ================================================

            elif name == "lock":

                arm.lock_current_orientation()

                print(
                    "Orientation locked."
                )

            # ================================================
            # FREE ORIENTATION (DEFAULT)
            # ================================================

            elif name == "free":

                arm.free_orientation()

                print(
                    "Orientation is free."
                )

            # ================================================
            # STOP
            # ================================================

            elif name == "stop":

                arm.stop()

            # ================================================
            # EXIT
            # ================================================

            elif name in (
                "quit",
                "exit",
            ):

                break

            else:

                print(
                    f"Unknown command: {name}"
                )

    finally:

        arm.shutdown()


if __name__ == "__main__":
    main()
