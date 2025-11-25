# IndoorRobot2025-v0

A small **indoor robot navigation** environment for Deep Reinforcement Learning.

Updated for **2025** with:

- Multiple corridor-like **scenarios** (train / test splits)
- **Static** and **dynamic** obstacles (moving blocks in corridors)
- Automatic **A* global path planning** with shortcut **smoothing**
- Robot must reach the goal while implicitly following the planned path and avoiding obstacles
- Several observation modes (pose only, pose + local map, global map)
- Clean 2D visualization using Matplotlib

The environment is implemented in `IndoorRobot2025Env` and registered as
`IndoorRobot2025-v0` with Gymnasium.

---

## 🔧 Installation

You need Python 3.9+.

```bash
git clone https://github.com/YOUR_USERNAME/IndoorRobot2025.git
cd IndoorRobot2025

pip install -r requirements.txt
pip install -e .
```

This will install the package `indoor_robot_2025` and register the environment
`IndoorRobot2025-v0`.

---

## 🧪 Quick Test

```python
import gymnasium as gym
import indoor_robot_2025   # just import to ensure registration

env = gym.make("IndoorRobot2025-v0", render_mode="human")
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

env.close()
```

---

## 📚 Environment Details

- **Action space**: `Discrete(3)`
  - `0`: go forward
  - `1`: go forward + turn left
  - `2`: go forward + turn right

- **Observation modes** (`observation_mode=`):
  - `ObservationMode.POSE_ONLY`:
    - `np.array([dx_unit, dy_unit, theta])`
  - `ObservationMode.POSE_AND_LOCAL_MAP`:
    - `(pose_vector, local_occupancy_grid)`
  - `ObservationMode.POSE_AND_LOCAL_INLINE`:
    - `np.concatenate([pose_vector, local_occupancy_grid.flatten()])`
  - `ObservationMode.GLOBAL_MAP`:
    - `(robot_pose, global_occupancy_grid)`

- **Grid**:
  - `grid_size` cells per side (default 64)
  - `grid_resolution` meters per cell (default 0.25)
  - Global occupancy:
    - `static_grid`: walls + structural obstacles
    - `dynamic_grid`: moving obstacles

- **Scenarios**:
  - 6 predefined corridor-like maps:
    - 4 for **training**: IDs `[0, 1, 2, 3]`
    - 2 for **testing**: IDs `[4, 5]`
  - Select split with `scenario_split="train"` or `"test"`

- **Planner**:
  - A* on the static grid in discrete space
  - Result is converted to world coordinates
  - Then a simple shortcut smoothing removes unnecessary waypoints
  - The smoothed path is available in `info["path_world"]` and drawn in the render

---

## 👩‍🏫 For Students

This repo is meant as a compact playground for:

- Classical planning vs Deep RL
- Navigation with partial observability (pose + local map)
- Obstacle avoidance with moving obstacles
- Training and evaluating RL agents across different scenarios / splits

You can plug this directly into standard Gymnasium-compatible RL code
(e.g. stable-baselines3, custom PyTorch training loops, etc.).

---

## 📄 License

MIT License.
