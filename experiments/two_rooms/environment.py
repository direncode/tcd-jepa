"""Two Rooms gridworld environment.

A simple navigation environment with two rooms connected by a doorway.
Generates RGB observation images for JEPA pretraining.
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TwoRoomsEnv:
    """Two-room gridworld with visual observations.

    Layout: two rectangular rooms connected by a single doorway.
    Agent navigates and the environment renders RGB observations.
    """

    ACTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # left, right, up, down

    def __init__(
        self,
        room_height: int = 5,
        room_width: int = 5,
        door_pos: int = 2,
        render_size: int = 64,
    ) -> None:
        self.room_height = room_height
        self.room_width = room_width
        self.door_pos = door_pos
        self.render_size = render_size

        self.grid_height = room_height
        self.grid_width = room_width * 2 + 1  # Two rooms + wall column
        self.wall_col = room_width  # Column index of the dividing wall

        self._build_grid()
        self.agent_pos = [1, 1]

    def _build_grid(self) -> None:
        """Build the grid: 0=floor, 1=wall."""
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)

        # Outer walls
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Dividing wall with door
        self.grid[:, self.wall_col] = 1
        if 0 < self.door_pos < self.grid_height - 1:
            self.grid[self.door_pos, self.wall_col] = 0  # doorway

    def reset(self, room: Optional[int] = None) -> np.ndarray:
        """Reset agent to random position in specified room (0 or 1).

        Returns:
            RGB observation [render_size, render_size, 3].
        """
        if room is None:
            room = np.random.randint(2)

        if room == 0:
            valid_cols = range(1, self.wall_col)
        else:
            valid_cols = range(self.wall_col + 1, self.grid_width - 1)

        valid_rows = range(1, self.grid_height - 1)

        while True:
            r = np.random.choice(list(valid_rows))
            c = np.random.choice(list(valid_cols))
            if self.grid[r, c] == 0:
                self.agent_pos = [r, c]
                break

        return self.render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take an action (0=left, 1=right, 2=up, 3=down).

        Returns:
            (observation, reward, done, info)
        """
        dr, dc = self.ACTIONS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        # Check bounds and walls
        if (0 <= new_r < self.grid_height
                and 0 <= new_c < self.grid_width
                and self.grid[new_r, new_c] == 0):
            self.agent_pos = [new_r, new_c]

        obs = self.render()
        # Room detection
        in_room = 0 if self.agent_pos[1] < self.wall_col else 1

        return obs, 0.0, False, {"room": in_room, "pos": tuple(self.agent_pos)}

    def render(self) -> np.ndarray:
        """Render current state as RGB image.

        Returns:
            RGB array [render_size, render_size, 3] in [0, 255].
        """
        # Create color grid
        H, W = self.grid_height, self.grid_width
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # Floor colors: different per room
        for r in range(H):
            for c in range(W):
                if self.grid[r, c] == 1:
                    img[r, c] = [80, 80, 80]  # walls: gray
                elif c < self.wall_col:
                    img[r, c] = [200, 220, 255]  # room 0: light blue
                else:
                    img[r, c] = [255, 220, 200]  # room 1: light orange

        # Agent
        img[self.agent_pos[0], self.agent_pos[1]] = [255, 50, 50]  # red

        # Upscale to render_size
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((self.render_size, self.render_size), Image.NEAREST)
        return np.array(pil_img)

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
    """Dataset of Two Rooms observations for JEPA pretraining."""

    def __init__(
        self,
        num_episodes: int = 100,
        episode_length: int = 50,
        room_height: int = 5,
        room_width: int = 5,
        render_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.render_size = render_size
        rng = np.random.RandomState(seed)

        env = TwoRoomsEnv(
            room_height=room_height,
            room_width=room_width,
            render_size=render_size,
        )

        self.observations = []
        for ep in range(num_episodes):
            np.random.seed(rng.randint(2**31))
            traj = env.generate_random_trajectory(episode_length)
            self.observations.extend(traj)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns observation as [C, H, W] float tensor in [0, 1]."""
        obs = self.observations[idx]
        tensor = torch.from_numpy(obs).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        return tensor
