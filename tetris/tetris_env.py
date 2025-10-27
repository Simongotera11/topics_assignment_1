# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Dict, List
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TetrisEnv(gym.Env):
    """
    Tetris environment (feature-based, discrete actions).

    Observation (float32, shape [12]):
        [ agg_height, max_height, bumpiness, holes, score ] + one_hot(current_piece in [I,O,T,S,Z,J,L])

    Actions (Discrete(6)):
        0=left, 1=right, 2=rot+, 3=rot-, 4=soft drop, 5=hard drop

    Rendering:
        render_mode in {"human","rgb_array"} with metadata["render_fps"].
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    SHAPES: Dict[str, List[List[List[int]]]] = {
        'I': [
            [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
            [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
            [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],
            [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]],
        ],
        'O': [
            [[0,1,1,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]],
        ] * 4,
        'T': [
            [[0,1,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
            [[0,1,0,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
            [[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,0]],
            [[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]],
        ],
        'S': [
            [[0,1,1,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]],
            [[0,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]],
            [[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]],
            [[1,0,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]],
        ],
        'Z': [
            [[1,1,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]],
            [[0,0,1,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
            [[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]],
            [[0,1,0,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]],
        ],
        'J': [
            [[1,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
            [[0,1,1,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
            [[0,0,0,0],[1,1,1,0],[0,0,1,0],[0,0,0,0]],
            [[0,1,0,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]],
        ],
        'L': [
            [[0,0,1,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
            [[0,1,0,0],[0,1,0,0],[0,1,1,0],[0,0,0,0]],
            [[0,0,0,0],[1,1,1,0],[1,0,0,0],[0,0,0,0]],
            [[1,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
        ],
    }
    SHAPE_NAMES = list(SHAPES.keys())

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        reward_mode: str = "hybrid",
        max_steps: int = 10000,
        drop_speed: int = 30,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.drop_speed = int(drop_speed)  # gravity ticks; smaller => faster
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Observation: 5 aggregates + 7 one-hot piece = 12
        high = np.array([200.0, 20.0, 200.0, 200.0, 1e6] + [1.0]*7, dtype=np.float32)
        low  = np.array([0.0,   0.0,  0.0,   0.0,  0.0 ] + [0.0]*7, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        # Renderer (lazy init)
        self._pygame = None
        self._pg_ready = False
        self._pg_window = None
        self._pg_clock = None
        self._cell = 24
        self._margin = 2

        self.reset(seed=seed)

    # ---------------- Core Gym Methods ----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rnd.seed(seed)
            self._np_rng = np.random.default_rng(seed)

        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.int8)
        self.current_piece = self._spawn_piece()
        self.current_rotation = 0
        self.current_x = self.BOARD_WIDTH // 2 - 2
        self.current_y = 0

        self.steps = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.score = 0
        self.game_over = False
        self.drop_counter = 0
        # NOTE: do NOT overwrite self.drop_speed here; respect constructor/eval setting

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        if self.game_over:
            return self._get_obs(), 0.0, True, False, self._get_info()

        reward = 0.0
        old_lines = self.lines_cleared
        old_holes = self._count_holes()
        old_bump  = self._get_bumpiness()
        old_col_h = self._get_column_heights()
        old_agg_h = int(old_col_h.sum())

        moved_horiz = False
        piece_locked = False

        # ---- ACTIONS ----
        if action == 0 and self._can_move(self.current_x - 1, self.current_y, self.current_rotation):
            self.current_x -= 1
            moved_horiz = True
        elif action == 1 and self._can_move(self.current_x + 1, self.current_y, self.current_rotation):
            self.current_x += 1
            moved_horiz = True
        elif action == 2:  # rot+
            nr = (self.current_rotation + 1) % 4
            if self._can_move(self.current_x, self.current_y, nr):
                self.current_rotation = nr
        elif action == 3:  # rot-
            nr = (self.current_rotation - 1) % 4
            if self._can_move(self.current_x, self.current_y, nr):
                self.current_rotation = nr
        elif action == 4:  # soft drop
            if self._can_move(self.current_x, self.current_y + 1, self.current_rotation):
                self.current_y += 1
                reward += 0.05
        elif action == 5:  # hard drop
            drop = 0
            while self._can_move(self.current_x, self.current_y + 1, self.current_rotation):
                self.current_y += 1
                drop += 1
            reward += drop * 0.05
            self._lock_piece()
            self._spawn_next_piece()
            piece_locked = True

        # ---- GRAVITY ----
        self.drop_counter += 1
        if self.drop_counter >= self.drop_speed:
            self.drop_counter = 0
            if self._can_move(self.current_x, self.current_y + 1, self.current_rotation):
                self.current_y += 1
            else:
                self._lock_piece()
                self._spawn_next_piece()
                piece_locked = True

        self.steps += 1

        # ---- REWARD FUNCTION  ----
        # Keep line-clear spikes
        cleared_now = self.lines_cleared - old_lines
        if cleared_now > 0:
            reward += [0, 100, 300, 700, 1500][min(cleared_now, 4)]

        # Tiny nudges to encourage movement/rotation exploration
        if moved_horiz: reward += 0.05
        if action in (2, 3): reward += 0.05  # rotations

        holes_now = self._count_holes()
        bump_now  = self._get_bumpiness()
        col_h_now = self._get_column_heights()
        agg_h_now = int(col_h_now.sum())
        max_h_now = self._get_max_height()

        # Potential phi(s) = -(w_h*holes + w_b*bump + w_a*agg_height)
        if piece_locked:
            phi_old = -(10.0 * old_holes + 0.5 * old_bump + 0.25 * old_agg_h)
            phi_now = -(10.0 * holes_now + 0.5 * bump_now + 0.25 * agg_h_now)
            reward += 1.0 * (phi_now - phi_old)   # improvement only at lock time
            reward -= 0.10 * max_h_now            # gentle landing-height penalty

        # NO per-step structural penalties here (avoids reward sinking with long episodes)

        # Terminal
        if self.game_over:
            reward -= 100.0

        truncated = self.steps >= self.max_steps
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human()

        return obs, float(reward), bool(self.game_over), bool(truncated), info

    # ---------------- Helpers ----------------

    def _spawn_piece(self) -> str:
        return self._rnd.choice(self.SHAPE_NAMES)

    def _spawn_next_piece(self):
        self.pieces_placed += 1
        self.current_piece = self._spawn_piece()
        self.current_rotation = 0
        self.current_x = self.BOARD_WIDTH // 2 - 2
        self.current_y = 0
        if not self._can_move(self.current_x, self.current_y, self.current_rotation):
            self.game_over = True

    def _can_move(self, x: int, y: int, rotation: int) -> bool:
        shape = self.SHAPES[self.current_piece][rotation]
        for i in range(4):
            for j in range(4):
                if shape[i][j]:
                    bx, by = x + j, y + i
                    if bx < 0 or bx >= self.BOARD_WIDTH or by >= self.BOARD_HEIGHT:
                        return False
                    if by >= 0 and self.board[by, bx]:
                        return False
        return True

    def _lock_piece(self):
        shape = self.SHAPES[self.current_piece][self.current_rotation]
        pid = self.SHAPE_NAMES.index(self.current_piece) + 1
        for i in range(4):
            for j in range(4):
                if shape[i][j]:
                    bx, by = self.current_x + j, self.current_y + i
                    if 0 <= bx < self.BOARD_WIDTH and 0 <= by < self.BOARD_HEIGHT:
                        self.board[by, bx] = pid
        self._clear_lines()

    def _clear_lines(self):
        lines = [y for y in range(self.BOARD_HEIGHT) if np.all(self.board[y, :] > 0)]
        if lines:
            self.lines_cleared += len(lines)
            self.score += [0, 100, 300, 500, 800][min(len(lines), 4)]
            for y in reversed(lines):
                self.board = np.delete(self.board, y, axis=0)
                self.board = np.vstack([np.zeros((1, self.BOARD_WIDTH), dtype=np.int8), self.board])

    def _get_max_height(self) -> int:
        for y in range(self.BOARD_HEIGHT):
            if np.any(self.board[y, :] > 0):
                return self.BOARD_HEIGHT - y
        return 0

    def _get_column_heights(self) -> np.ndarray:
        heights = np.zeros(self.BOARD_WIDTH, dtype=np.int32)
        for x in range(self.BOARD_WIDTH):
            for y in range(self.BOARD_HEIGHT):
                if self.board[y, x] > 0:
                    heights[x] = self.BOARD_HEIGHT - y
                    break
        return heights

    def _count_holes(self) -> int:
        holes = 0
        for x in range(self.BOARD_WIDTH):
            seen_block = False
            for y in range(self.BOARD_HEIGHT):
                if self.board[y, x] > 0:
                    seen_block = True
                elif seen_block and self.board[y, x] == 0:
                    holes += 1
        return holes

    def _get_bumpiness(self) -> int:
        h = self._get_column_heights()
        return int(np.sum(np.abs(np.diff(h))))

    # ---------------- Observation & Info ----------------

    def _get_obs(self) -> np.ndarray:
        col_h = self._get_column_heights()
        agg_height = float(col_h.sum())
        max_h = float(col_h.max())
        bump = float(np.sum(np.abs(np.diff(col_h))))
        holes = float(self._count_holes())

        onehot = np.zeros(7, dtype=np.float32)
        onehot[self.SHAPE_NAMES.index(self.current_piece)] = 1.0

        obs = np.array([agg_height, max_h, bump, holes, float(self.score)], dtype=np.float32)
        return np.concatenate([obs, onehot], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "lines_cleared": int(self.lines_cleared),
            "pieces_placed": int(self.pieces_placed),
            "max_height": int(self._get_max_height()),
            "holes": int(self._count_holes()),
            "score": int(self.score),
            "current_y": int(self.current_y),
        }

    # ---------------- Rendering ----------------

    def _ensure_pygame(self):
        if self._pg_ready:
            return
        try:
            import pygame  # lazy import
        except ImportError as e:
            raise ImportError("pygame is required for human rendering. pip install pygame") from e
        self._pygame = __import__("pygame")
        self._pg_clock = self._pygame.time.Clock()
        self._pg_ready = True
        self._pg_window = None
        # Colors
        self._bg_color = (15, 15, 18)
        self._grid_color = (40, 40, 50)
        self._colors = {
            "I": (0, 180, 200), "O": (230, 200, 0), "T": (160, 0, 200),
            "S": (0, 180, 0), "Z": (200, 0, 0), "J": (0, 0, 200), "L": (220, 120, 0),
        }

    def _open_window(self):
        self._ensure_pygame()
        pg = self._pygame
        w = self.BOARD_WIDTH * (self._cell + self._margin) + self._margin
        h = self.BOARD_HEIGHT * (self._cell + self._margin) + self._margin
        if self._pg_window is None:
            pg.display.init()
            self._pg_window = pg.display.set_mode((w, h))
            pg.display.set_caption("Tetris (Gymnasium)")

    def _blit_board(self, surface):
        pg = self._pygame
        surface.fill(self._bg_color)

        # copy settled board
        temp = np.array(self.board, copy=True)

        # overlay falling piece
        shape = np.array(self.SHAPES[self.current_piece][self.current_rotation], dtype=np.int32)
        for dy in range(4):
            for dx in range(4):
                if shape[dy, dx]:
                    by = self.current_y + dy
                    bx = self.current_x + dx
                    if 0 <= by < self.BOARD_HEIGHT and 0 <= bx < self.BOARD_WIDTH:
                        temp[by, bx] = -1  # sentinel for falling piece

        # draw cells
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                val = temp[r, c]
                if val == 0:
                    color = (22, 22, 28)
                elif val == -1:
                    color = self._colors.get(self.current_piece, (180, 180, 180))
                else:
                    color = list(self._colors.values())[(val - 1) % 7]
                x = self._margin + c * (self._cell + self._margin)
                y = self._margin + r * (self._cell + self._margin)
                pg.draw.rect(surface, color, pg.Rect(x, y, self._cell, self._cell))
                pg.draw.rect(surface, self._grid_color, pg.Rect(x, y, self._cell, self._cell), 1)

    def _render_human(self):
        self._open_window()
        pg = self._pygame
        self._blit_board(self._pg_window)
        pg.display.flip()

        # process events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.game_over = True

        fps = int(self.metadata.get("render_fps", 10))
        self._pg_clock.tick(max(fps, 1))

    def render(self):
        if self.render_mode == "human":
            self._render_human()
            return None
        elif self.render_mode == "rgb_array":
            self._ensure_pygame()
            w = self.BOARD_WIDTH * (self._cell + self._margin) + self._margin
            h = self.BOARD_HEIGHT * (self._cell + self._margin) + self._margin
            surf = self._pygame.Surface((w, h))
            self._blit_board(surf)
            arr = self._pygame.surfarray.array3d(surf)  # WxHx3
            return np.transpose(arr, (1, 0, 2))  # HxWx3
        return None

    def close(self):
        try:
            super().close()
        except Exception:
            pass
        if self._pg_ready and self._pygame is not None:
            try:
                self._pygame.display.quit()
            except Exception:
                pass
        self._pg_window = None
