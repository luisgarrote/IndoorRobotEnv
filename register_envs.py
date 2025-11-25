import gymnasium as gym
from .indoor_robot_2025 import IndoorRobot2025Env

gym.register(
    id="IndoorRobot2025-v0",
    entry_point="indoor_robot_2025:IndoorRobot2025Env",
)
