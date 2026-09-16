import time

from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower


config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="iw_so101",
)

robot = SO101Follower(config)

print("Connecting...")
robot.connect()

print("Connected.")

# Read current joint positions
obs = robot.get_observation()

print("\nCurrent positions:")
for name, value in obs.items():
    print(name, value)

print("\nTorque should now be ON.")
print("Try VERY GENTLY moving the joints by hand.")
print("They should resist movement.")

time.sleep(10)

print("\nDisconnecting...")
robot.disconnect()

print("Done.")
