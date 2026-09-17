import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


class TestNode(Node):

    def __init__(self):
        super().__init__("so101_pan_test")

        self.positions = {}

        self.create_subscription(
            JointState,
            "/follower/joint_states",
            self.cb,
            10,
        )

        self.client = ActionClient(
            self,
            FollowJointTrajectory,
            "/follower/arm_trajectory_controller/follow_joint_trajectory",
        )

    def cb(self, msg):
        self.positions = dict(
            zip(msg.name, msg.position)
        )


rclpy.init()

node = TestNode()

# Ждём joint states
deadline = time.time() + 5

while (
    time.time() < deadline
    and not all(j in node.positions for j in JOINTS)
):
    rclpy.spin_once(
        node,
        timeout_sec=0.1,
    )

if not all(j in node.positions for j in JOINTS):
    raise RuntimeError("No joint states")

print("\nBEFORE:")

for j in JOINTS:
    print(
        f"{j:15s} = "
        f"{node.positions[j]:+.4f}"
    )

before = [
    node.positions[j]
    for j in JOINTS
]

target = before.copy()

# ТОЛЬКО shoulder_pan
target[0] += 0.10

print("\nTARGET:")

for j, value in zip(
    JOINTS,
    target,
):
    print(
        f"{j:15s} = "
        f"{value:+.4f}"
    )

if not node.client.wait_for_server(
    timeout_sec=5.0
):
    raise RuntimeError(
        "Trajectory controller unavailable"
    )

goal = FollowJointTrajectory.Goal()

goal.trajectory.joint_names = JOINTS

point = JointTrajectoryPoint()

point.positions = target

point.time_from_start.sec = 2

goal.trajectory.points = [
    point
]

future = node.client.send_goal_async(
    goal
)

rclpy.spin_until_future_complete(
    node,
    future,
)

handle = future.result()

if not handle.accepted:
    raise RuntimeError("Goal rejected")

result_future = (
    handle.get_result_async()
)

rclpy.spin_until_future_complete(
    node,
    result_future,
    timeout_sec=5.0,
)

print(
    "\nACTION RESULT:",
    result_future.result()
)

# обновить состояние
end = time.time() + 1.0

while time.time() < end:
    rclpy.spin_once(
        node,
        timeout_sec=0.1,
    )

print("\nAFTER:")

for j in JOINTS:
    print(
        f"{j:15s} = "
        f"{node.positions[j]:+.4f}"
    )

print(
    "\nshoulder_pan delta =",
    node.positions["shoulder_pan"]
    - before[0],
)

node.destroy_node()
rclpy.shutdown()