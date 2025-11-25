import gymnasium as gym
#import indoor_robot_2025_animated  # ensures env is registered

from indoor_robot_2025_animated import IndoorRobot2025Env

gym.register(
    id="IndoorRobot2025-v0",
    entry_point="indoor_robot_2025_animated:IndoorRobot2025Env",
)

env = gym.make("IndoorRobot2025-v0", render_mode="human")
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

env.close()
