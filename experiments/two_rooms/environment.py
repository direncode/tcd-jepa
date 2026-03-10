"""Multi-room gridworld environment.

A navigation environment with configurable room layouts and corridors.
Generates RGB observation images for JEPA pretraining. The topological
structure (rooms, corridors, dead-ends) provides natural module candidates.
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TwoRoomsEnv:
    """Multi-room gridworld with visual observations.

    Default: 4 rooms in a 2x2 grid connected by narrow corridors.
    The corridor structure creates natural topological features:
    - Each room is an attractor (H_0)
    - Corridors connecting rooms form cycles if traversed (H_1)
    - Room boundaries are topological boundaries (H_2)
    """

    ACTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # left, right, up, down

    def __init__(
        self,
        room_height: int = 7,
        room_width: int = 7,
        num_rooms_x: int = 2,
        num_rooms_y: int = 2,
        corridor_width: int = 1,
        render_size: int = 64,
        add_objects: bool = True,
    ) -> None:
        self.room_height = room_height
        self.room_width = room_width
        self.num_rooms_x = num_rooms_x
        self.num_rooms_y = num_rooms_y
        self.corridor_width = corridor_width
        self.render_size = render_size
        self.add_objects = add_objects

        self.grid_height = num_rooms_y * room_height + (num_rooms_y - 1)
        self.grid_width = num_rooms_x * room_width + (num_rooms_x - 1)

        self._build_grid()
        self.agent_pos = [1, 1]

        # Room-specific colors for visual diversity
        self._room_colors = [
            [200, 220, 255],  # light blue
            [255, 220, 200],  # light orange
            [220, 255, 220],  # light green
            [255, 220, 255],  # light pink
            [255, 255, 200],  # light yellow
            [220, 240, 255],  # light cyan
        ]

        # Place static objects for visual complexity
        self._objects = []
        if add_objects:
            self._place_objects()

    def _build_grid(self) -> None:
        """Build multi-room grid: 0=floor, 1=wall."""
        H, W = self.grid_height, self.grid_width
        self.grid = np.ones((H, W), dtype=np.int32)
        self._room_map = np.full((H, W), -1, dtype=np.int32)

        room_id = 0
        for ry in range(self.num_rooms_y):
            for rx in range(self.num_rooms_x):
                top = ry * (self.room_height + 1)
                left = rx * (self.room_width + 1)
                # Carve room interior
                for r in range(top, min(top + self.room_height, H)):
                    for c in range(left, min(left + self.room_width, W)):
                        self.grid[r, c] = 0
                        self._room_map[r, c] = room_id
                room_id += 1

        # Carve corridors between rooms
        for ry in range(self.num_rooms_y):
            for rx in range(self.num_rooms_x):
                # Horizontal corridor to the right
                if rx < self.num_rooms_x - 1:
                    wall_col = rx * (self.room_width + 1) + self.room_width
                    mid_row = ry * (self.room_height + 1) + self.room_height // 2
                    for dr in range(self.corridor_width):
                        r = mid_row + dr
                        if 0 <= r < H:
                            self.grid[r, wall_col] = 0
                            self._room_map[r, wall_col] = -2  # corridor

                # Vertical corridor downward
                if ry < self.num_rooms_y - 1:
                    wall_row = ry * (self.room_height + 1) + self.room_height
                    mid_col = rx * (self.room_width + 1) + self.room_width // 2
                    for dc in range(self.corridor_width):
                        c = mid_col + dc
                        if 0 <= c < W:
                            self.grid[wall_row, c] = 0
                            self._room_map[wall_row, c] = -2  # corridor

    def _place_objects(self) -> None:
        """Place colored objects in rooms for visual variety."""
        rng = np.random.RandomState(42)
        object_colors = [
            [50, 50, 200],   # blue
            [50, 200, 50],   # green
            [200, 200, 50],  # yellow
            [200, 50, 200],  # magenta
        ]
        for room_id in range(self.num_rooms_x * self.num_rooms_y):
            room_cells = list(zip(*np.where(self._room_map == room_id)))
            if len(room_cells) > 4:
                n_objects = rng.randint(1, 4)
                chosen = rng.choice(len(room_cells), min(n_objects, len(room_cells)), replace=False)
                for idx in chosen:
                    r, c = room_cells[idx]
                    color = object_colors[room_id % len(object_colors)]
                    self._objects.append((r, c, color))

    def reset(self, room: Optional[int] = None) -> np.ndarray:
        """Reset agent to random position."""
        n_rooms = self.num_rooms_x * self.num_rooms_y
        if room is None:
            room = np.random.randint(n_rooms)
        room = room % n_rooms

        room_cells = list(zip(*np.where(self._room_map == room)))
        # Filter out object positions
        obj_positions = {(r, c) for r, c, _ in self._objects}
        free_cells = [(r, c) for r, c in room_cells if (r, c) not in obj_positions]
        if not free_cells:
            free_cells = room_cells

        idx = np.random.randint(len(free_cells))
        self.agent_pos = list(free_cells[idx])
        return self.render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take an action."""
        dr, dc = self.ACTIONS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        if (0 <= new_r < self.grid_height
                and 0 <= new_c < self.grid_width
                and self.grid[new_r, new_c] == 0):
            self.agent_pos = [new_r, new_c]

        obs = self.render()
        in_room = self._room_map[self.agent_pos[0], self.agent_pos[1]]
        return obs, 0.0, False, {"room": int(in_room), "pos": tuple(self.agent_pos)}

    def render(self) -> np.ndarray:
        """Render current state as RGB image."""
        H, W = self.grid_height, self.grid_width
        img = np.zeros((H, W, 3), dtype=np.uint8)

        for r in range(H):
            for c in range(W):
                if self.grid[r, c] == 1:
                    img[r, c] = [60, 60, 60]  # walls: dark gray
                elif self._room_map[r, c] >= 0:
                    room_id = self._room_map[r, c]
                    img[r, c] = self._room_colors[room_id % len(self._room_colors)]
                else:
                    img[r, c] = [180, 180, 180]  # corridors: light gray

        # Draw objects
        for r, c, color in self._objects:
            img[r, c] = color

        # Agent
        img[self.agent_pos[0], self.agent_pos[1]] = [255, 50, 50]  # red

        # Upscale to render_size using numpy (no Pillow dependency)
        scale_r = self.render_size / H
        scale_c = self.render_size / W
        upscaled = np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)
        for r in range(self.render_size):
            for c in range(self.render_size):
                src_r = min(int(r / scale_r), H - 1)
                src_c = min(int(c / scale_c), W - 1)
                upscaled[r, c] = img[src_r, src_c]

        return upscaled

    def generate_random_trajectory(
        self,
        length: int = 100,
        room: Optional[int] = None,
    ) -> list[np.ndarray]:
        """Generate a random walk trajectory of observations."""
        observations = [self.reset(room=room)]
        for _ in range(length - 1):
            action = np.random.randint(4)
            obs, _, _, _ = self.step(action)
            observations.append(obs)
        return observations


class TwoRoomsDataset(Dataset):
    """Dataset of multi-room observations for JEPA pretraining.

    Generates diverse trajectories across all rooms to create rich
    topological structure in the learned representations.
    """

    def __init__(
        self,
        num_episodes: int = 100,
        episode_length: int = 50,
        room_height: int = 7,
        room_width: int = 7,
        num_rooms_x: int = 2,
        num_rooms_y: int = 2,
        render_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.render_size = render_size
        rng = np.random.RandomState(seed)

        env = TwoRoomsEnv(
            room_height=room_height,
            room_width=room_width,
            num_rooms_x=num_rooms_x,
            num_rooms_y=num_rooms_y,
            render_size=render_size,
        )
        n_rooms = num_rooms_x * num_rooms_y

        self.observations = []
        for ep in range(num_episodes):
            np.random.seed(rng.randint(2**31))
            # Alternate starting rooms to get cross-room trajectories
            start_room = ep % n_rooms
            traj = env.generate_random_trajectory(episode_length, room=start_room)
            self.observations.extend(traj)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns observation as [C, H, W] float tensor in [0, 1]."""
        obs = self.observations[idx]
        tensor = torch.from_numpy(obs).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        return tensor
