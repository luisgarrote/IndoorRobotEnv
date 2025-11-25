import math
from enum import Enum
from typing import Tuple, Dict, Any, List, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class ObservationMode(Enum):
    POSE = 1
    DIRECTION = 2
    POSE_AND_ERROR = 3
    DIRECTION_AND_ERROR = 4
    LOCAL=5


 




class IndoorRobot2025Env(gym.Env):
    """
    IndoorRobot2025-v0

    A 2D indoor navigation environment for Deep RL with:
      - Multiple corridor-like scenarios (train/test splits)
      - Static and dynamic obstacles
      - Start/goal regions per scenario
      - A* global path planning with simple smoothing
      - The RL agent controls robot velocity/steering and must reach the goal
        while implicitly following the planned path and avoiding obstacles.

    Observation modes:
      - POSE_ONLY:
          [dx_unit, dy_unit, theta]  (unit vector to goal in robot frame + heading)
      - POSE_AND_LOCAL_MAP:
          (pose_like_above, local_occupancy_grid)
      - POSE_AND_LOCAL_INLINE:
          [pose_like_above, local_occupancy_grid.flatten()]
      - GLOBAL_MAP:
          (pose, global_occupancy_grid)

    Action space (Discrete 3):
      0: go forward
      1: go forward + steer left
      2: go forward + steer right
    """

    metadata = {"render_modes": ["human", "rgb_array","ignore_mode"], "render_fps": 20}

    def __init__(
        self,
        grid_size: int = 64,
        local_grid_size: int = 21,
        grid_resolution: float = 0.25,
        observation_mode: ObservationMode = ObservationMode.LOCAL,
        render_mode: Optional[str] = "rgb_array",
        scenario_split: str = "train",  # "train" or "test"
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self._static_background_cache = None

        assert grid_size >= 16 and grid_size % 2 == 0, "grid_size must be >=16 and even"
        assert local_grid_size % 2 == 1, "local_grid_size should be odd (for symmetry)"
        assert scenario_split in ("train", "test")

        self.grid_size = grid_size
        self.local_grid_size = local_grid_size
        self.grid_resolution = float(grid_resolution)
        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.scenario_split = scenario_split

        # Gymnasium seeding
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Discrete steering actions
        self.action_space = spaces.Discrete(3)

        # We'll represent the robot pose as [x, y, theta]
        # but observation depends on mode
  
        self._dummy_obs_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(local_grid_size,local_grid_size),
            dtype=np.float32,
        )
        # Use generic Box for safety; agent code should know the mode it uses
        self.observation_space = self._dummy_obs_space

        # Global occupancy grids
        self.static_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.dynamic_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.heatmap = np.zeros((grid_size, grid_size), dtype=np.int32)

        self.local_grid = np.zeros((local_grid_size, local_grid_size), dtype=np.float32)

        self.robot_pose = np.zeros(3, dtype=np.float32)
        self.goal_position = np.zeros(2, dtype=np.float32)


        # Robot sprite for global view (RGB arrow icon)
        self._robot_sprite_base = self._create_robot_sprite(size=16)
        self.max_steps = 1000
        self.current_step = 0
        self.total_reward = 0.0

        # Global path (world coordinates) from A*
        self.path_world: List[Tuple[float, float]] = []
        self.smoothed_path_world: List[Tuple[float, float]] = []
        self._last_distance_to_goal = 0.0
        self._last_action = -1

        # Dynamic obstacles: each is dict with "path" (list of (i,j)) and "idx"
        self.dynamic_obstacles: List[Dict[str, Any]] = []

        # Scenario bookkeeping
        self.scenario_id: int = 0
        self._train_scenarios = [0, 1, 2, 3]
        self._test_scenarios = [4, 5]


        self.floor_tex = self._gen_floor_texture()
        self.wall_tex = self._gen_wall_texture()
        # self.robot_sprites = [
        #     self._gen_robot_sprite(direction=k) for k in range(4)
        # ]
        # Render state
        self._last_rgb = None




        self.mapping = {
            0: np.array([0.1, 0.0], dtype=np.float32),
            1: np.array([0.1, +0.4], dtype=np.float32),
            2: np.array([0.1, -0.4], dtype=np.float32),
            3: np.array([0.1, +0.7], dtype=np.float32),
            4: np.array([0.1, -0.7], dtype=np.float32),
            5: np.array([0.2, 0.0], dtype=np.float32),
            6: np.array([0.2, +0.4], dtype=np.float32),
            7: np.array([0.2, -0.4], dtype=np.float32),
            8: np.array([0.2, +0.7], dtype=np.float32),
            9: np.array([0.2, -0.7], dtype=np.float32),
        }



        self.action_space = spaces.Discrete(len(self.mapping))





    def _compute_visible_walls(self):
        """
        Computes which wall cells are 'exterior' (touch outside space)
        using a flood-fill from outside map border.
        Returns a boolean mask of shape (grid_size, grid_size).
        """
        N = self.grid_size
        visited = np.zeros((N, N), dtype=np.uint8)
        queue = []

        # Start flood-fill from border free cells
        for i in range(N):
            for j in [0, N-1]:
                if self.static_grid[i, j] == 0:
                    queue.append((i, j))
                    visited[i, j] = 1
        for j in range(N):
            for i in [0, N-1]:
                if self.static_grid[i, j] == 0:
                    queue.append((i, j))
                    visited[i, j] = 1

        # BFS through free space
        while queue:
            i, j = queue.pop()
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < N and 0 <= nj < N:
                    if visited[ni, nj] == 0 and self.static_grid[ni, nj] == 0:
                        visited[ni, nj] = 1
                        queue.append((ni, nj))

        # A wall is exterior if any neighbor is in visited=1 (outside air)
        ext = np.zeros((N, N), dtype=np.uint8)
        walls = np.where(self.static_grid == 1)
        for i, j in zip(walls[0], walls[1]):
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < N and 0 <= nj < N:
                    if visited[ni, nj] == 1:
                        ext[i, j] = 1
                        break
        return ext
    # ----------------------------------------------------------------------
    # Simple textures (procedural) for global rendering
    # ----------------------------------------------------------------------
    def _gen_floor_texture(self,size=8):
        tex = np.zeros((size, size, 3), dtype=np.float32)
        # checkerboard tile
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    tex[i, j] = np.array([0.92, 0.92, 0.95])
                else:
                    tex[i, j] = np.array([0.88, 0.88, 0.92])
        return tex

    def _gen_wall_texture(self,size=8):
        tex = np.zeros((size, size, 3), dtype=np.float32)
        # brick-like speckled wall
        base = np.array([0.85, 0.85, 0.88])
        noise = np.random.uniform(-0.05, 0.05, (size, size, 3))
        tex[:] = base + noise
        return np.clip(tex, 0, 1)

    # def _gen_robot_sprite(self,size=8, direction=0):
    #     """
    #     direction: 0=right, 1=up, 2=left, 3=down
    #     Produces a small arrow sprite.
    #     """
    #     img = np.zeros((size, size, 3))
    #     cx = cy = size // 2
    #     for y in range(size):
    #         for x in range(size):
    #             dx = x - cx
    #             dy = y - cy
    #             r = math.sqrt(dx*dx + dy*dy)
    #             if r > size/2: 
    #                 continue
    #             # arrow body
    #             if abs(dy) < 2 and dx > -2:
    #                 img[y, x] = [0.2, 0.7, 0.2]
    #             # arrow head facing the direction
    #             if direction == 0 and dx > 1 and abs(dy) < 3:
    #                 img[y, x] = [0.1, 0.5, 0.1]
    #             if direction == 2 and dx < -1 and abs(dy) < 3:
    #                 img[y, x] = [0.1, 0.5, 0.1]
    #             if direction == 1 and dy < -1 and abs(dx) < 3:
    #                 img[y, x] = [0.1, 0.5, 0.1]
    #             if direction == 3 and dy > 1 and abs(dx) < 3:
    #                 img[y, x] = [0.1, 0.5, 0.1]
    #     return img




    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Robot sprite helper for global rendering
    # ------------------------------------------------------------------
    def _create_robot_sprite(self, size: int = 16) -> np.ndarray:
        """
        Create a simple arrow-shaped sprite pointing along +x in local
        sprite coordinates. We will rotate it in the renderer according
        to the robot's theta using an Affine2D transform.
        """
        img = np.zeros((size, size, 4), dtype=np.float32)
        cx = size // 2
        cy = size // 2
        radius = size / 2.2

        for y in range(size):
            for x in range(size):
                dx = x - cx
                dy = y - cy
                r = math.sqrt(dx * dx + dy * dy)
                if r > radius:
                    continue

                # Body: light green
                if abs(dy) < radius * 0.25 and dx > -radius * 0.2:
                    img[y, x, :3] = [0.3, 0.8, 0.3]
                    img[y, x, 3] = 1.0

                # Head: darker green pointing to +x
                if dx > radius * 0.15 and abs(dy) < radius * 0.35:
                    img[y, x, :3] = [0.1, 0.5, 0.1]
                    img[y, x, 3] = 1.0

        # soft alpha at edges
        alpha = img[:, :, 3]
        img[:, :, 3] = alpha
        return img

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world (x,y) to grid indices (i,j)."""
        j = int(round(x / self.grid_resolution + self.grid_size / 2.0))
        i = int(round(y / self.grid_resolution + self.grid_size / 2.0))
        i = np.clip(i, 0, self.grid_size - 1)
        j = np.clip(j, 0, self.grid_size - 1)
        return i, j

    def _grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        """Convert grid indices (i,j) to world (x,y)."""
        x = (j - self.grid_size / 2.0) * self.grid_resolution
        y = (i - self.grid_size / 2.0) * self.grid_resolution
        return float(x), float(y)

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------
    def _build_scenario(self, scenario_id: int):
        """
        Build a corridor-like scenario by carving rectangular free spaces
        inside a fully occupied map.
        """
        # Start with solid walls everywhere
        self.static_grid[:, :] = 1
        self.dynamic_grid[:, :] = 0
        self.dynamic_obstacles.clear()

        N = self.grid_size
        c = N // 2

        def carve_rect(i_min, i_max, j_min, j_max):
            self.static_grid[i_min:i_max + 1, j_min:j_max + 1] = 0

        # Default: outer boundary already walls (static_grid==1)
        # Scenarios:
        #   0: straight vertical corridor + left branch
        #   1: straight vertical corridor + right branch
        #   2: S-corridor (two turns)
        #   3: plus-shaped intersection
        #   4,5: more narrow test scenarios

        if scenario_id == 0:
            # Main vertical corridor
            carve_rect(4, N - 5, c - 2, c + 2)
            # Left branch near middle
            mid = c + (N // 6)
            carve_rect(mid - 2, mid + 2, 4, c + 2)
            # Start region (bottom of main corridor)
            start_region = (5, 10, c - 1, c + 1)  # i_min, i_max, j_min, j_max
            # Goal region (end of left branch)
            goal_region = (mid - 1, mid + 1, 5, 10)
            dyn_segments = [((mid, c - 1), (mid, c + 1))]

        elif scenario_id == 1:
            # Main vertical corridor
            carve_rect(4, N - 5, c - 2, c + 2)
            # Right branch
            mid = c + (N // 6)
            carve_rect(mid - 2, mid + 2, c - 2, N - 5)
            start_region = (5, 10, c - 1, c + 1)
            goal_region = (mid - 1, mid + 1, N - 10, N - 5)
            dyn_segments = [((mid, c - 1), (mid, c + 1))]

        elif scenario_id == 2:
            # S-shaped corridor
            # lower vertical
            carve_rect(4, c, c - 2, c + 2)
            # middle horizontal to the right
            carve_rect(c - 2, c + 2, c, N - 5)
            # upper vertical going up
            carve_rect(c, N - 5, N - 7, N - 3)

            start_region = (5, 10, c - 1, c + 1)
            goal_region = (N - 10, N - 6, N - 6, N - 4)
            dyn_segments = [((c, c + 3), (c, N - 8))]

        elif scenario_id == 3:
            # plus intersection
            carve_rect(4, N - 5, c - 2, c + 2)     # vertical
            carve_rect(c - 2, c + 2, 4, N - 5)     # horizontal

            start_region = (5, 10, c - 1, c + 1)
            goal_region = (N - 10, N - 6, c - 1, c + 1)
            dyn_segments = [((c, c + 5), (c, N - 8)),
                            ((c, 4), (c, c - 5))]

        elif scenario_id == 4:
            # narrow zig-zag (test)
            width = 2
            carve_rect(4, c, c - width, c + width)
            carve_rect(c - width, c + width, c - 10, c + 10)
            carve_rect(c, N - 5, c - width, c + width)

            start_region = (5, 8, c - 1, c + 1)
            goal_region = (N - 10, N - 6, c - 1, c + 1)
            dyn_segments = [((c, c - 1), (c, c + 1))]

        else:  # scenario_id == 5
            # U-shaped corridor (test)
            carve_rect(4, N - 5, c - 2, c + 2)
            carve_rect(4, 12, c + 2, N - 5)
            carve_rect(N - 12, N - 5, c + 2, N - 5)

            start_region = (5, 10, c - 1, c + 1)
            goal_region = (N - 10, N - 6, N - 8, N - 5)
            dyn_segments = [((c + 6, 4), (c + 6, N - 8))]

        # Sample start and goal in free cells
        start_ij = self._sample_free_cell(start_region)
        goal_ij = self._sample_free_cell(goal_region)
        sx, sy = self._grid_to_world(*start_ij)
        gx, gy = self._grid_to_world(*goal_ij)

        self.robot_pose = np.array([sx, sy, 0.0], dtype=np.float32)
        self.goal_position = np.array([gx, gy], dtype=np.float32)

        # Initialize heatmap
        self.heatmap[:, :] = 0

        # Build dynamic obstacles paths from segments in grid space
        self.dynamic_obstacles.clear()
        for (p0, p1) in dyn_segments:
            path = self._bresenham_line(p0[0], p0[1], p1[0], p1[1])
            if len(path) > 1:
                self.dynamic_obstacles.append({"path": path, "idx": 0, "dir": 1})

        # Plan a global path with A* on static_grid only
        self._plan_and_smooth_path(start_ij, goal_ij)

    def _sample_free_cell(self, region: Tuple[int, int, int, int]) -> Tuple[int, int]:
        i_min, i_max, j_min, j_max = region
        for _ in range(2000):
            i = self.np_random.integers(i_min, i_max + 1)
            j = self.np_random.integers(j_min, j_max + 1)
            if self.static_grid[i, j] == 0:
                return int(i), int(j)
        # fallback: any free cell
        free = np.argwhere(self.static_grid == 0)
        if len(free) == 0:
            raise RuntimeError("Scenario has no free cells")
        idx = self.np_random.integers(0, len(free))
        i, j = free[idx]
        return int(i), int(j)

    # ------------------------------------------------------------------
    # A* + smoothing
    # ------------------------------------------------------------------
    # def _plan_and_smooth_path(self, start_ij: Tuple[int, int], goal_ij: Tuple[int, int]):
    #     path_cells = self._astar(start_ij, goal_ij)
    #     if not path_cells:
    #         # if planner fails, just set straight-line world path for visualization
    #         sx, sy = self._grid_to_world(*start_ij)
    #         gx, gy = self._grid_to_world(*goal_ij)
    #         self.path_world = [(sx, sy), (gx, gy)]
    #         self.smoothed_path_world = list(self.path_world)
    #         return

    #     # convert to world coords
    #     self.path_world = [self._grid_to_world(i, j) for (i, j) in path_cells]
    #     # simple shortcut smoothing
    #     smoothed = []
    #     if len(path_cells) <= 2:
    #         smoothed = self.path_world
    #     else:
    #         i0 = 0
    #         smoothed.append(self.path_world[0])
    #         while i0 < len(path_cells) - 1:
    #             i1 = len(path_cells) - 1
    #             while i1 > i0 + 1:
    #                 if self._line_free(path_cells[i0], path_cells[i1]):
    #                     break
    #                 i1 -= 1
    #             smoothed.append(self._grid_to_world(*path_cells[i1]))
    #             i0 = i1
    #     self.smoothed_path_world = smoothed



    def _plan_and_smooth_path(self, start_ij, goal_ij):
        # ------------------------------------------------------------
        # 1) Run A*
        # ------------------------------------------------------------
        path_cells = self._astar(start_ij, goal_ij)
        if not path_cells:
            sx, sy = self._grid_to_world(*start_ij)
            gx, gy = self._grid_to_world(*goal_ij)
            self.path_world = [(sx, sy), (gx, gy)]
            self.smoothed_path_world = list(self.path_world)
            return

        # raw world path
        self.path_world = [self._grid_to_world(i, j) for (i, j) in path_cells]
        pts = np.array(self.path_world, dtype=float)

        # trivial short paths
        if len(pts) <= 2:
            self.smoothed_path_world = list(self.path_world)
            return

        # ------------------------------------------------------------
        # 2) Elastic Bands smoothing
        # ------------------------------------------------------------

        # parameters
        tension_k = 0.4         # how strongly the path tries to become straight
        repulse_k = 2.5          # how strongly it avoids obstacles
        repulse_radius = 0.6     # meters
        step_size = 0.2          # integration step
        iters = 40               # number of smoothing iterations

        # convert grid to world geometry access
        N = self.grid_size
        res = self.grid_resolution

        def dist_to_obstacle(x, y):
            """Distance to nearest wall in world coordinates."""
            # convert world -> grid
            gi = int((y / res) + N/2)
            gj = int((x / res) + N/2)
            if gi < 0 or gi >= N or gj < 0 or gj >= N:
                return 0.0  # outside = obstacle

            # if inside a wall, zero distance
            if self.static_grid[gi, gj] == 1:
                return 0.0

            # sample neighbors (small radius)
            dmin = 10.0
            R = int(repulse_radius / res) + 2
            for di in range(-R, R+1):
                for dj in range(-R, R+1):
                    ni = gi + di
                    nj = gj + dj
                    if 0 <= ni < N and 0 <= nj < N and self.static_grid[ni, nj] == 1:
                        wx, wy = self._grid_to_world(ni, nj)
                        d = math.hypot(x - wx, y - wy)
                        dmin = min(dmin, d)
            return dmin

        p = pts.copy()   # working copy

        for _ in range(iters):
            for k in range(1, len(p)-1):  # endpoints fixed
                x, y = p[k]

                # ----- tension force (pulls toward midpoint of neighbors)
                prev = p[k-1]
                nextp = p[k+1]
                midpoint = 0.5 * (prev + nextp)
                tension_force = tension_k * (midpoint - p[k])

                # ----- repulsive force from walls
                d = dist_to_obstacle(x, y)
                if d < repulse_radius and d > 1e-6:
                    # direction away from nearest wall
                    # approximate gradient by finite differences
                    eps = 0.05
                    dx = dist_to_obstacle(x+eps, y) - d
                    dy = dist_to_obstacle(x, y+eps) - d
                    grad = np.array([dx, dy])
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 0:
                        grad /= grad_norm
                    repulse_force = repulse_k * (repulse_radius - d) * grad
                else:
                    repulse_force = np.zeros(2)

                # ----- update point
                p[k] += step_size * (tension_force + repulse_force)

        # save smoothed
        self.smoothed_path_world = [(float(x), float(y)) for x, y in p]









    def _neighbors(self, i: int, j: int):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                if self.static_grid[ni, nj] == 0:  # free
                    yield ni, nj

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        import heapq

        (si, sj) = start
        (gi, gj) = goal

        def h(i, j):
            return abs(i - gi) + abs(j - gj)

        open_set = []
        heapq.heappush(open_set, (0 + h(si, sj), 0, (si, sj)))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {(si, sj): 0}

        while open_set:
            _, g, (i, j) = heapq.heappop(open_set)
            if (i, j) == (gi, gj):
                # reconstruct
                path = [(i, j)]
                while (i, j) in came_from:
                    i, j = came_from[(i, j)]
                    path.append((i, j))
                path.reverse()
                return path

            for (ni, nj) in self._neighbors(i, j):
                tentative_g = g + 1
                if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                    g_score[(ni, nj)] = tentative_g
                    came_from[(ni, nj)] = (i, j)
                    f = tentative_g + h(ni, nj)
                    heapq.heappush(open_set, (f, tentative_g, (ni, nj)))

        return []

    def _bresenham_line(self, i0, j0, i1, j1):
        """Return list of (i,j) cells along Bresenham line."""
        points = []
        dx = abs(j1 - j0)
        dy = abs(i1 - i0)
        x, y = j0, i0
        sx = 1 if j1 > j0 else -1
        sy = 1 if i1 > i0 else -1
        if dx > dy:
            err = dx / 2.0
            while x != j1:
                points.append((y, x))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != i1:
                points.append((y, x))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((i1, j1))
        return points

    def _line_free(self, p0: Tuple[int, int], p1: Tuple[int, int]) -> bool:
        """Check if straight line between p0 and p1 crosses any walls (static_grid)."""
        for (i, j) in self._bresenham_line(p0[0], p0[1], p1[0], p1[1]):
            if self.static_grid[i, j] != 0:
                return False
        return True

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    @property
    def info(self) -> Dict[str, Any]:
        return {
            "goal": {"x": float(self.goal_position[0]), "y": float(self.goal_position[1])},
            "robot": {
                "x": float(self.robot_pose[0]),
                "y": float(self.robot_pose[1]),
                "theta": float(self.robot_pose[2]),
            },
            "scenario_id": int(self.scenario_id),
            "path_world": self.smoothed_path_world,
            "last_action": int(self._last_action),
            "total_reward": float(self.total_reward),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)


        self._static_background_cache = None

        self.current_step = 0
        self.total_reward = 0.0
        self._last_action = -1
        self._last_rgb = None
        
        # choose scenario
        if self.scenario_split == "train":
            pool = self._train_scenarios
        else:
            pool = self._test_scenarios

        self.scenario_id = int(self.np_random.choice(pool))
        self._build_scenario(self.scenario_id)

        self.wall_visibility = self._compute_visible_walls()

        # initialize distance to goal
        self._last_distance_to_goal = np.linalg.norm(
            self.robot_pose[:2] - self.goal_position
        )

        # build local grid
        self._update_local_grid()

        obs = self._build_observation()
        return obs, self.info

    def set_goal(self, goal_xy: Tuple[float, float]):
        """Manually set goal in world coordinates and recompute path."""
        gx, gy = goal_xy
        self.goal_position = np.array([gx, gy], dtype=np.float32)
        start_ij = self._world_to_grid(self.robot_pose[0], self.robot_pose[1])
        goal_ij = self._world_to_grid(self.goal_position[0], self.goal_position[1])
        self._plan_and_smooth_path(start_ij, goal_ij)

    def step(self, action: int):
        self._last_action = int(action)
        if action < 0:
            # sometimes useful for rendering still frames
            return self._build_observation(), 0.0, False, False, self.info

        # Update dynamic obstacles first (so we collide with new positions)
        self._update_dynamic_obstacles()

        # Action mapping
        # [linear_velocity, angular_velocity]

        v, w = self.mapping[action]

        # Simple unicycle integration
        h = 1.0
        x, y, th = self.robot_pose
        dx = h * v * math.cos(th)
        dy = h * v * math.sin(th)
        dth = h * w
        th_new = math.atan2(math.sin(th + dth), math.cos(th + dth))
        x_new = x + dx
        y_new = y + dy

        # bounds in world coordinates
        limit = self.grid_resolution * self.grid_size / 2.0
        terminated = False
        collision_penalty = 0.0

        if (-limit <= x_new < limit) and (-limit <= y_new < limit):
            self.robot_pose = np.array([x_new, y_new, th_new], dtype=np.float32)
        else:
            terminated = True
            collision_penalty = -10.0

        # grid collision check with static + dynamic obstacles
        i, j = self._world_to_grid(self.robot_pose[0], self.robot_pose[1])
        self.heatmap[i, j] += 1
        if self.static_grid[i, j] != 0 or self.dynamic_grid[i, j] != 0:
            terminated = True
            collision_penalty = -10.0

        # Distance-based shaping
        dist = float(np.linalg.norm(self.robot_pose[:2] - self.goal_position))
        progress = self._last_distance_to_goal - dist
        self._last_distance_to_goal = dist

        reached_goal = dist < 0.25
        if reached_goal and not terminated:
            terminated = True

        # path following bonus (encourage staying close to smoothed path)
        path_error = self._distance_to_path(self.robot_pose[0], self.robot_pose[1])
        reward = 0.0
        reward += 1.5 * progress          # forward progress to goal
        reward -= 0.01                     # time penalty
        reward -= 0.2 * path_error         # penalize deviation from planned path
        reward += collision_penalty
        if reached_goal:
            reward += 10.0

        self.total_reward += reward
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # Update local grid
        self._update_local_grid()

        obs = self._build_observation()
        return obs, reward, terminated, truncated, self.info



    def _compute_lateral_error_point(self, robot_pose, path_world, point_goal):
        """
        Compute lateral (cross-track) error between robot pose and a path.
        path_world: list of (x, y) in world coordinates.

        Returns:
            lateral_error (float): signed distance from robot to path
                                   positive = robot is left of path direction
                                   negative = robot is right of path direction
        """
        if path_world is None or len(path_world) < 2:
            return 0.0

        rx, ry, rtheta = robot_pose

        #--- step 1: find closest segment on the path ----------------------
        min_dist = float("inf")
        closest_point = None
        segment_direction = None

        for i in range(len(path_world) - 1):
            x1, y1 = path_world[i]
            x2, y2 = path_world[i + 1]

            # vector from point1 to robot
            vx = rx - x1
            vy = ry - y1

            # segment direction
            sx = x2 - x1
            sy = y2 - y1
            seg_len2 = sx*sx + sy*sy
            if seg_len2 < 1e-9:
                continue

            # projection scalar (0 = near x1,y1; 1 = near x2,y2)
            t = max(0.0, min(1.0, (vx*sx + vy*sy) / seg_len2))

            # projected point
            px = x1 + t * sx
            py = y1 + t * sy

            # distance to robot
            d = math.hypot(px - rx, py - ry)

            if d < min_dist:
                min_dist = d
                closest_point = (px, py)
     

        if closest_point is None:
            return point_goal

        return closest_point


    def _compute_lateral_error(self, robot_pose, path_world):
        """
        Compute lateral (cross-track) error between robot pose and a path.
        path_world: list of (x, y) in world coordinates.

        Returns:
            lateral_error (float): signed distance from robot to path
                                   positive = robot is left of path direction
                                   negative = robot is right of path direction
        """
        if path_world is None or len(path_world) < 2:
            return 0.0

        rx, ry, rtheta = robot_pose

        #--- step 1: find closest segment on the path ----------------------
        min_dist = float("inf")
        closest_point = None
        segment_direction = None

        for i in range(len(path_world) - 1):
            x1, y1 = path_world[i]
            x2, y2 = path_world[i + 1]

            # vector from point1 to robot
            vx = rx - x1
            vy = ry - y1

            # segment direction
            sx = x2 - x1
            sy = y2 - y1
            seg_len2 = sx*sx + sy*sy
            if seg_len2 < 1e-9:
                continue

            # projection scalar (0 = near x1,y1; 1 = near x2,y2)
            t = max(0.0, min(1.0, (vx*sx + vy*sy) / seg_len2))

            # projected point
            px = x1 + t * sx
            py = y1 + t * sy

            # distance to robot
            d = math.hypot(px - rx, py - ry)

            if d < min_dist:
                min_dist = d
                closest_point = (px, py)
                segment_direction = (sx, sy)

        if closest_point is None:
            return 0.0

        px, py = closest_point
        sx, sy = segment_direction

        #--- step 2: compute signed lateral error --------------------------

        # vector from closest point → robot
        dx = rx - px
        dy = ry - py

        # normalize path direction
        seg_mag = math.hypot(sx, sy)
        if seg_mag < 1e-9:
            return 0.0
        tx = sx / seg_mag
        ty = sy / seg_mag

        # normal pointing "left" of the segment
        nx = -ty
        ny =  tx

        # signed distance
        lateral = dx * nx + dy * ny
        return lateral
    # ------------------------------------------------------------------
    # Path-distance helper
    # ------------------------------------------------------------------
    def _distance_to_path(self, x: float, y: float) -> float:
        if not self.smoothed_path_world:
            return 0.0
        px = np.array([p[0] for p in self.smoothed_path_world])
        py = np.array([p[1] for p in self.smoothed_path_world])
        dx = px - x
        dy = py - y
        dists = np.sqrt(dx * dx + dy * dy)
        return float(np.min(dists))

    # ------------------------------------------------------------------
    # Dynamic obstacles update
    # ------------------------------------------------------------------
    def _update_dynamic_obstacles(self):
        # reset dynamic grid
        self.dynamic_grid[:, :] = 0
        for dobj in self.dynamic_obstacles:
            path = dobj["path"]
            idx = dobj["idx"]
            direction = dobj["dir"]
            if len(path) == 0:
                continue
            # current position
            i, j = path[idx]
            self.dynamic_grid[i, j] = 1
            # advance index (back-and-forth motion)
            next_idx = idx + direction
            if next_idx < 0 or next_idx >= len(path):
                direction *= -1
                next_idx = idx + direction
            dobj["idx"] = next_idx
            dobj["dir"] = direction

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    # def _update_local_grid(self):
    #     """Rebuild local occupancy grid centered on the robot."""
    #     L = self.local_grid_size
    #     self.local_grid[:, :] = 0.0

    #     i_center, j_center = self._world_to_grid(self.robot_pose[0], self.robot_pose[1])
    #     half = L // 2
    #     for di in range(-half, half + 1):
    #         for dj in range(-half, half + 1):
    #             gi = i_center + di
    #             gj = j_center + dj
    #             li = di + half
    #             lj = dj + half
    #             if 0 <= gi < self.grid_size and 0 <= gj < self.grid_size:
    #                 val = self.static_grid[gi, gj] or self.dynamic_grid[gi, gj]
    #                 self.local_grid[li, lj] = float(val)

    #     # mark goal cell with -1 if within local window
    #     gi_goal, gj_goal = self._world_to_grid(self.goal_position[0], self.goal_position[1])
    #     di_goal = gi_goal - i_center
    #     dj_goal = gj_goal - j_center
    #     if abs(di_goal) <= half and abs(dj_goal) <= half:
    #         li = di_goal + half
    #         lj = dj_goal + half
    #         self.local_grid[li, lj] = -1.0
    def _update_local_grid(self):
        """Rebuild local occupancy grid in the ROBOT's local (egocentric) frame."""
        L = self.local_grid_size
        res = self.grid_resolution
        self.local_grid[:, :] = 0.0

        # robot world pose
        rx, ry, th = self.robot_pose

        # half window size
        half = L // 2

        # ---------------------------------------------------------
        # 1) Fill static+dynamic world cells rotated into robot frame
        # ---------------------------------------------------------
        for li in range(L):
            for lj in range(L):
                # local cell coordinate in robot frame
                dx = (li - half + 0.5) * res     # +x = forward
                dy = (lj - half + 0.5) * res     # +y = left

                # rotate robot->world (+theta)
                wx = rx + dx * math.cos(th) - dy * math.sin(th)
                wy = ry + dx * math.sin(th) + dy * math.cos(th)

                # convert to world grid
                gi, gj = self._world_to_grid(wx, wy)

                if 0 <= gi < self.grid_size and 0 <= gj < self.grid_size:
                    val = self.static_grid[gi, gj] or self.dynamic_grid[gi, gj]
                    self.local_grid[li, lj] = float(val)

        # ---------------------------------------------------------
        # 2) Mark PATH in egocentric space
        # ---------------------------------------------------------
        if hasattr(self, "smoothed_path_world") and self.smoothed_path_world:
            for (xw, yw) in self.smoothed_path_world:
                # convert path world→robot LOCAL frame
                dx = xw - rx
                dy = yw - ry

                # rotate world->robot (-theta)
                lx = dx * math.cos(-th) - dy * math.sin(-th)
                ly = dx * math.sin(-th) + dy * math.cos(-th)

                # convert local meters → local grid
                li = int(lx / res + half)
                lj = int(ly / res + half)

                if 0 <= li < L and 0 <= lj < L and self.local_grid[li, lj] <= 0:
                    self.local_grid[li, lj] = -0.5

        # ---------------------------------------------------------
        # 3) Mark GOAL in egocentric space
        # ---------------------------------------------------------
        gx, gy = self.goal_position

        dx = gx - rx
        dy = gy - ry
        lx = dx * math.cos(-th) - dy * math.sin(-th)
        ly = dx * math.sin(-th) + dy * math.cos(-th)

        li = int(lx / res + half)
        lj = int(ly / res + half)
        if 0 <= li < L and 0 <= lj < L:
            self.local_grid[li, lj] = -1.0


    # def _update_local_grid(self):
    #     """Rebuild local occupancy grid centered on the robot."""
    #     L = self.local_grid_size
    #     self.local_grid[:, :] = 0.0

    #     i_center, j_center = self._world_to_grid(self.robot_pose[0], self.robot_pose[1])
    #     half = L // 2

    #     # ---------------------------------------------------------
    #     # 1) Copy static + dynamic grids into local grid
    #     # ---------------------------------------------------------
    #     for di in range(-half, half + 1):
    #         for dj in range(-half, half + 1):
    #             gi = i_center + di
    #             gj = j_center + dj
    #             li = di + half
    #             lj = dj + half
    #             if 0 <= gi < self.grid_size and 0 <= gj < self.grid_size:
    #                 val = self.static_grid[gi, gj] or self.dynamic_grid[gi, gj]
    #                 self.local_grid[li, lj] = float(val)

    #     # ---------------------------------------------------------
    #     # 2) Mark the path as -0.5 (local representation)
    #     # ---------------------------------------------------------
    #     if hasattr(self, "smoothed_path_world") and self.smoothed_path_world:
    #         for (xw, yw) in self.smoothed_path_world:
    #             # convert world→grid
    #             gi, gj = self._world_to_grid(xw, yw)

    #             di = gi - i_center
    #             dj = gj - j_center

    #             # only if inside the local window
    #             if abs(di) <= half and abs(dj) <= half:
    #                 li = di + half
    #                 lj = dj + half

    #                 # Do not overwrite walls / obstacles (positive values)
    #                 if self.local_grid[li, lj] <= 0:
    #                     self.local_grid[li, lj] = -0.5

    #     # ---------------------------------------------------------
    #     # 3) Mark goal cell with -1 if within local window
    #     # ---------------------------------------------------------
    #     gi_goal, gj_goal = self._world_to_grid(self.goal_position[0], self.goal_position[1])
    #     di_goal = gi_goal - i_center
    #     dj_goal = gj_goal - j_center
    #     if abs(di_goal) <= half and abs(dj_goal) <= half:
    #         li = di_goal + half
    #         lj = dj_goal + half
    #         self.local_grid[li, lj] = -1.0

    def _build_observation(self):
        # vector to goal in world frame
        dx = self.goal_position[0] - self.robot_pose[0]
        dy = self.goal_position[1] - self.robot_pose[1]
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8
        dx_n = dx / dist
        dy_n = dy / dist
        th = float(self.robot_pose[2])

        pose_vec = np.array([dx_n, dy_n, th], dtype=np.float32)

        if self.observation_mode == ObservationMode.POSE:
            return np.array([self.robot_pose[0], self.robot_pose[1], self.robot_pose[2]], dtype=np.float32)
        elif self.observation_mode == ObservationMode.DIRECTION:
            return pose_vec
        elif self.observation_mode == ObservationMode.POSE_AND_ERROR:
            error=self._compute_lateral_error(self.robot_pose,self.smoothed_path_world)
            return np.array([self.robot_pose[0], self.robot_pose[1], self.robot_pose[2], error], dtype=np.float32)
        elif self.observation_mode == ObservationMode.DIRECTION_AND_ERROR:
            error=self._compute_lateral_error(self.robot_pose,self.smoothed_path_world)
            return np.array([dx_n, dy_n, th, error], dtype=np.float32)
        elif self.observation_mode == ObservationMode.LOCAL:
            return self.local_grid.flatten()


        # default fallback
        return pose_vec

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):

        # lets consider a ignore_mode for fast simulation

        if self.render_mode == "human":
 
            fig, (ax, ax2) = plt.subplots(1, 2,figsize=(4, 4))
            #ax = plt.gca()
            self._draw_scene(ax)
            #plt.title(f"IndoorRobot2025 - scenario {self.scenario_id}")
            self._draw_scene_local(ax2)
            self._draw_minimap(ax2)

            plt.show()
            #plt.pause(0.001)   # allows GUI event loop to update
 
            return None

        if self.render_mode == "rgb_array":
            if self._last_action == -1 and self._last_rgb is not None:
                return self._last_rgb

            plt.switch_backend("Agg")
            fig, (ax, ax2) = plt.subplots(1, 2,figsize=(4, 4))
            self._draw_scene(ax)

            self._draw_scene_local(ax2)
            self._draw_minimap(ax2)

            fig.canvas.draw()
            #image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            rgba = np.asarray(fig.canvas.buffer_rgba())   # modern API
            image = rgba[..., :3]   
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            self._last_rgb = image
            return image



    def _draw_minimap(self, ax):
        """
        Draws the local robot-centric map as a minimap in the corner
        of the global scene. Uses inset axes so it behaves like a HUD.
        """
        # Size of the minimap (fraction of main plot)
        mini_ax = inset_axes(ax,
                             width="30%",   # % of parent
                             height="30%",
                             loc='upper right',
                             borderpad=1.0)

        # Remove tick labels
        mini_ax.set_xticks([])
        mini_ax.set_yticks([])
        mini_ax.set_aspect('equal')

        # prepare bounds
        L = self.local_grid_size
        res = self.grid_resolution
        half = L * res / 2

        # Draw background
        mini_ax.set_xlim(-half, half)
        mini_ax.set_ylim(-half, half)
        mini_ax.set_facecolor((0.95, 0.95, 0.95))

        # ------------------------------------------
        # Draw local grid (same as _draw_scene_local but simplified)
        # ------------------------------------------
        for i in range(L):
            for j in range(L):
                v = self.local_grid[i, j]
                wx = (i - L/2) * res
                wy = (j - L/2) * res

                if v > 0:                # obstacle https://colorhunt.co/palette/b7e5cd8abeb9305669c1785a
                    color = "#B7E5CD"
                    fill = True
                    lw = 0
                elif v < -0.75:          # goal
                    color = "#8ABEB9"
                    fill = True
                    lw = 0
                elif v < -0.25:          # path
                    color = "#305669"
                    fill = True
                    lw = 0
                else:
                    color = "#C1785A"
                    fill = True
                    lw = 0

                mini_ax.add_patch(
                    plt.Rectangle((wx, wy), res, res,
                                  facecolor=color if fill else "none",
                                  edgecolor=color,
                                  linewidth=lw)
                )

        # # draw robot at center
        # mini_ax.add_patch(plt.Circle((0, 0), 0.1, color='blue'))

        # # heading arrow
        # th = self.robot_pose[2]
        # mini_ax.plot([0, 0.2*math.cos(th)],
        #              [0, 0.2*math.sin(th)],
        #              color='red', linewidth=2)

        return mini_ax




    def _world_to_robot_local(self, xw, yw):
        """
        Convert a world coordinate (xw, yw) into robot-local frame:
            robot at (0,0) facing +x in local frame.
        """
        rx, ry, th = self.robot_pose

        # translate
        dx = xw - rx
        dy = yw - ry

        # rotate by -theta
        lx =  dx * math.cos(-th) - dy * math.sin(-th)
        ly =  dx * math.sin(-th) + dy * math.cos(-th)

        return lx, ly


    def generate_local_grid(self):
        """Rebuild local occupancy grid centered on the robot."""
        L = self.local_grid_size
        temp_grid= self.local_grid*0.0

        i_center, j_center = self._world_to_grid(self.robot_pose[0], self.robot_pose[1])
        half = L // 2
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                gi = i_center + di
                gj = j_center + dj
                li = di + half
                lj = dj + half
                if 0 <= gi < self.grid_size and 0 <= gj < self.grid_size:
                    val = self.static_grid[gi, gj] or self.dynamic_grid[gi, gj]
                    temp_grid[li, lj] = float(val)

        # mark goal cell with -1 if within local window
        gi_goal, gj_goal = self._world_to_grid(self.goal_position[0], self.goal_position[1])
        di_goal = gi_goal - i_center
        dj_goal = gj_goal - j_center
        if abs(di_goal) <= half and abs(dj_goal) <= half:
            li = di_goal + half
            lj = dj_goal + half
            temp_grid[li, lj] = -1.0
        return temp_grid


    def _draw_scene_local(self, ax):
        

        temp_grid=self.local_grid

        ax.axis([-self.grid_resolution*self.local_grid_size/2,self.grid_resolution*self.local_grid_size/2,-self.grid_resolution*self.local_grid_size/2,self.grid_resolution*self.local_grid_size/2])
        ax.axis('square')



        for i in range(self.local_grid_size):
            for j in range(self.local_grid_size):
                if(temp_grid[j][i]>0):
                    ax.add_artist( plt.Rectangle(((j-self.local_grid_size/2)*self.grid_resolution, (i-self.local_grid_size/2)*self.grid_resolution), self.grid_resolution, self.grid_resolution, fill=True, color='grey', linewidth=2))
                elif(temp_grid[j][i]<-0.5):

                    ax.add_artist( plt.Rectangle(((j-self.local_grid_size/2)*self.grid_resolution, (i-self.local_grid_size/2)*self.grid_resolution), self.grid_resolution, self.grid_resolution, fill=False, color='black', linewidth=0.1))


        if hasattr(self, "smoothed_path_world") and self.smoothed_path_world:
            path_local_x = []
            path_local_y = []

            half = self.grid_resolution * self.local_grid_size / 2.0

            for (xw, yw) in self.smoothed_path_world:
                lx, ly = self._world_to_robot_local(xw, yw)

                # include only points inside local window
                if -half <= lx <= half and -half <= ly <= half:
                    path_local_x.append(lx)
                    path_local_y.append(ly)

            # Draw the visible segment
            if len(path_local_x) > 1:
                ax.plot(path_local_x, path_local_y, color='#1f77b4', linewidth=2)


        ax.add_artist(plt.Circle((0, 0), 0.1, color='blue'))
        N = self.grid_size
        res = self.grid_resolution
        dy, dx = np.where(self.dynamic_grid == 1)
        for i in range(len(dx)):
            #print("On dx")

            px = (dx[i] - N/2) * res
            py = (dy[i] - N/2) * res
            #print(px,py)

            px, py = self._world_to_robot_local(px, py)

            #print(dx,dy)

            #print(px,py)
            #ax.scatter(px, py, c="#d62728", s=50, marker="s", edgecolor="black")
            ax.add_artist(plt.Circle((px, py),  res/2.0, color="#d62728"))




        if self._last_action>=0:

            v, w = self.mapping[self._last_action]


            x_arr=[0]
            y_arr=[0]

            x=0
            y=0
            theta=0
            for i in range(50):
                x=x+v*0.1*math.cos(theta)
                y=y+v*0.1*math.sin(theta)
                theta=theta+w*0.1
                x_arr.append(x)
                y_arr.append(y)
            
            ax.plot(x_arr, y_arr, color='#5a855f', linewidth=1)


            #print(self.robot_pose[2])
            x=[0,0+0.2*math.cos(0.0)]
            y=[0,0+0.2*math.sin(0.0)]
            ax.plot(x, y, color='#ff3333', linewidth=1)









            closest=self._compute_lateral_error_point(self.robot_pose+[0.2*math.cos(self.robot_pose[2]),0.2*math.sin(self.robot_pose[2]),0],self.smoothed_path_world,self.goal_position)

            
            lx, ly = self._world_to_robot_local(closest[0], closest[1])
            

            
            ax.plot([x[1], lx],[y[1], ly],"--", color='red', linewidth=1)

        plt.xticks(fontsize=10)
        plt.text(-1, -3, r'Total Reward: '+str(round(self.total_reward , 3)), fontsize=10)

        ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
 

    def _draw_scene(self, ax):
        limit = self.grid_resolution * self.grid_size / 2.0
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        N = self.grid_size
        res = self.grid_resolution
        if self._static_background_cache is None:
            # create an offscreen figure to render static tiles
            fig2, ax2 = plt.subplots(figsize=(4,4), dpi=100)
            ax2.set_xlim(-limit, limit)
            ax2.set_ylim(-limit, limit)
            ax2.set_aspect("equal")
            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.axis('off')
            fig2.subplots_adjust(bottom = 0)
            fig2.subplots_adjust(top = 1)
            fig2.subplots_adjust(right = 1)
            fig2.subplots_adjust(left = 0)
            # draw static layers ONCE:
            # - floor texture
            # - exterior walls
            # - interior walls
            # (your existing code for these stays the same)
            self._draw_static_layers(ax2)

            # convert to RGBA image
            fig2.canvas.draw()
            width, height = fig2.canvas.get_width_height()
            rgba = np.frombuffer(fig2.canvas.tostring_argb(), dtype=np.uint8)
            rgba = rgba.reshape((height, width, 4))
            rgba = rgba[:, :, [1,2,3,0]]  # ARGB → RGBA

            plt.close(fig2)
            self._static_background_cache = rgba
        
        #print("wall count:", np.sum(self.static_grid == 1))

        ax.imshow(
            self._static_background_cache,
            extent=(-limit, limit, -limit, limit),
            origin='upper',
            zorder=0
        )

        # # draw floor texture everywhere (background)
        # N = self.grid_size
        # res = self.grid_resolution
        # # tile draws: use imshow block
        # texture = self.floor_tex
        # ax.imshow(
        #     np.tile(texture, (N // texture.shape[0] + 1, N // texture.shape[1] + 1, 1))[:N,:N],
        #     extent=(-limit, limit, -limit, limit),
        #     origin="lower"
        # )

        # # draw exterior walls with texture
        # ext = self.wall_visibility
        # # wi, wj = np.where(ext == 1)
        # # if len(wi) > 0:
        # #     #print("On Wi")

        # #     px = (wj - N/2) * res
        # #     py = (wi - N/2) * res
        # #     ax.scatter(px, py, 
        # #                c=[self.wall_tex[0,0]]*len(px),
        # #                s=36, marker="s", edgecolor="black",
        # #                linewidth=0.3, alpha=1.0)

        # wi, wj = np.where(ext == 1)
        # tex = self.wall_tex                         # (8x8x3)

        # for i, j in zip(wi, wj):
        #     # Convert grid → world bottom-left corner
        #     wx = (j - N/2) * res
        #     wy = (i - N/2) * res

        #     # Draw single texture tile stretched to cell size
        #     ax.imshow(
        #         tex,
        #         extent=(wx, wx + res, wy, wy + res),
        #         origin="lower",
        #         interpolation="nearest",
        #         zorder=3
        #     )

        # # draw interior walls (darker)
        # # wi2, wj2 = np.where((self.static_grid == 1) & (ext == 0))
        # # if len(wi2) > 0:
        # #    # print("On Wi2")

        # #     px = (wj2 - N/2) * res
        # #     py = (wi2 - N/2) * res
        # #     ax.scatter(px, py,
        # #                c="#353535", s=30, marker="s",
        # #                alpha=0.8)
        # wi2, wj2 = np.where((self.static_grid == 1) & (ext == 0))

        # # generate interior-wall texture (darker)
        # interior_tex = np.clip(self.wall_tex * 0.55, 0, 1)   # darker version

        # for i, j in zip(wi2, wj2):
        #     wx = (j - N/2) * res
        #     wy = (i - N/2) * res

        #     ax.imshow(
        #         interior_tex,
        #         extent=(wx, wx + res, wy, wy + res),
        #         origin="lower",
        #         interpolation="nearest",
        #         zorder=3
        #     )

        # dynamic obstacles
 
        dy, dx = np.where(self.dynamic_grid == 1)
        for i in range(len(dx)):
            #print("On dx")

            px = (dx[i] - N/2) * res
            py = (dy[i] - N/2) * res
            #ax.scatter(px, py, c="#d62728", s=50, marker="s", edgecolor="black")
            ax.add_artist(plt.Circle((px, py),  res/2.0, color="#d62728"))


        # smoothed path (overlay)

        #print(self.smoothed_path_world)
        if len(self.smoothed_path_world) > 1:
            sx, sy = zip(*self.smoothed_path_world)
            ax.plot(sx, sy, "-", color="#1f77b4", linewidth=3, alpha=0.9)

        if len(self.path_world) > 1:
            sx, sy = zip(*self.path_world)
            ax.plot(sx, sy, "--", color="#ff77b4", linewidth=3, alpha=0.9)


        # robot sprite
        rx, ry, th = self.robot_pose
        # select sprite direction
        direction = 0
        if abs(th) < math.pi/4:         direction = 0   # right
        elif th >= math.pi/4 and th < 3*math.pi/4: direction = 1   # up
        elif abs(th) > 3*math.pi/4:     direction = 2   # left
        else:                            direction = 3   # down



        ax.add_artist(plt.Circle((self.robot_pose[0], self.robot_pose[1]), 0.1, color='blue'))
        x=[self.robot_pose[0],self.robot_pose[0]+0.2*math.cos(self.robot_pose[2])]
        y=[self.robot_pose[1],self.robot_pose[1]+0.2*math.sin(self.robot_pose[2])]
        ax.plot(x, y, color='#ff3333', linewidth=1)


        # goal
        gx, gy = self.goal_position
        ax.scatter([gx], [gy], c="#ffa600", s=120, marker="*", zorder=11, edgecolor="black")

        ax.set_xticks([])
        ax.set_yticks([])


        if self._last_action>=0:
            v, w = self.mapping[self._last_action]


            x_arr=[self.robot_pose[0]]
            y_arr=[self.robot_pose[1]]

            x=self.robot_pose[0]
            y=self.robot_pose[1]
            theta=self.robot_pose[2]
            for i in range(50):
                x=x+v*0.1*math.cos(theta)
                y=y+v*0.1*math.sin(theta)
                theta=theta+w*0.1
                x_arr.append(x)
                y_arr.append(y)
            
            ax.plot(x_arr, y_arr, color='#5a855f', linewidth=1)




    def _draw_static_layers(self, ax):
        """
        Draws only static (non-changing) grid elements:
          - floor texture
          - exterior walls (textured)
          - interior walls (textured)
        Called only once per episode.
        """
        N = self.grid_size
        res = self.grid_resolution
        ext = self.wall_visibility

        # ------- floor texture (one big imshow) -------
        texture = self.floor_tex
        floor_repeat_x = N * res
        floor_repeat_y = N * res

        ax.imshow(
            np.tile(texture, (N//texture.shape[0] + 1,
                              N//texture.shape[1] + 1, 1))[:N,:N],
            extent=(-floor_repeat_x/2, floor_repeat_x/2,
                    -floor_repeat_y/2, floor_repeat_y/2),
            origin='lower',
            interpolation="nearest",
            zorder=0
        )



        ext = np.zeros((N, N), dtype=np.uint8)
        for i, j in np.argwhere(self.static_grid == 1):
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                if 0 <= i+di < N and 0 <= j+dj < N:
                    if self.static_grid[i+di, j+dj] == 0:
                        ext[i,j] = 1
                        break
        # ------- EXTERIOR WALLS -------
        wi, wj = np.where(ext == 1)
        for i, j in zip(wi, wj):
            wx = (j - N/2) * res
            wy = (i - N/2) * res
            ax.imshow(
                self.wall_tex,
                extent=(wx, wx+res, wy, wy+res),
                origin='lower',
                interpolation='nearest',
                zorder=1
            )

        # ------- INTERIOR WALLS -------
        interior_tex = np.clip(self.wall_tex * 0.55, 0, 1)
        wi2, wj2 = np.where((self.static_grid == 1) & (ext == 0))
        for i, j in zip(wi2, wj2):
            wx = (j - N/2) * res
            wy = (i - N/2) * res
            ax.imshow(
                interior_tex,
                extent=(wx, wx+res, wy, wy+res),
                origin='lower',
                interpolation='nearest',
                zorder=1
            )




    def close(self):
        plt.close("all")
        super().close()


if __name__ == "__main__":
    # Manual quick test
    env = IndoorRobot2025Env(render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()
