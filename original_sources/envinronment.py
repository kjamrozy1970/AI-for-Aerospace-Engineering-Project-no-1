"""Continuous rover, discrete laser-coverage mask. No GUI dependency during RL."""
import math
import numpy as np
from matplotlib.path import Path
from new_rover import Rover


class Env:
    ACTIONS = Rover.ACTIONS

    def __init__(self, width=100, height=100, shape="circle", beam_width=15.0,
                 step_size=3.0, turn_angle_deg=15.0, max_steps=180,
                 goal_coverage=0.98, start=(50.0, 50.0, 0.0)):
        if width <= 0 or height <= 0 or beam_width <= 0 or step_size <= 0:
            raise ValueError("Map dimensions, beam width and forward step must be positive")
        if max_steps <= 0 or not 0 < goal_coverage <= 1:
            raise ValueError("max_steps must be positive and goal_coverage in (0, 1]")
        self.width = int(width)
        self.height = int(height)
        self.shape = shape
        self.beam_width = float(beam_width)
        self.max_steps = int(max_steps)
        self.goal_coverage = float(goal_coverage)
        self.start = tuple(start)
        if len(self.start) != 3 or not (0 <= self.start[0] <= width and 0 <= self.start[1] <= height):
            raise ValueError("Start pose must lie within the map")
        self.rover = Rover(*self.start, step_size=step_size, turn_angle_deg=turn_angle_deg)

        # Samples at pixel centres, one cell = 1 x 1 map unit.
        ys, xs = np.mgrid[0:self.height, 0:self.width]
        self.X = xs.astype(np.float32) + 0.5
        self.Y = ys.astype(np.float32) + 0.5

        if shape == "circle":
            self.center = (50.0, 50.0)
            self.radius = 25.0
            self.target = ((self.X - self.center[0]) ** 2 +
                           (self.Y - self.center[1]) ** 2 <= self.radius ** 2)
            self.vertices = None
        elif shape == "triangle":
            self.vertices = np.array([(20., 20.), (80., 20.), (50., 80.)])
            self.target = Path(self.vertices).contains_points(
                np.column_stack([self.X.ravel(), self.Y.ravel()])
            ).reshape(self.height, self.width)
        else:
            raise ValueError("shape must be 'circle' or 'triangle'")

        self.target_count = int(self.target.sum())
        assert self.target_count > 0
        self.reset()

    def _laser_footprint(self, x0, y0, x1, y1):
        """Capsule: circular laser beam swept continuously along a segment."""
        dx, dy = x1 - x0, y1 - y0
        d2 = dx * dx + dy * dy
        if d2 < 1e-12:
            sqdist = (self.X-x1)**2 + (self.Y-y1)**2
        else:
            t = np.clip(((self.X-x0)*dx + (self.Y-y0)*dy)/d2, 0, 1)
            sqdist = (self.X - (x0+t*dx))**2 + (self.Y-(y0+t*dy))**2
        return sqdist <= (self.beam_width / 2)**2

    @property
    def coverage_ratio(self):
        return float(np.count_nonzero(self.scanned & self.target) / self.target_count)

    def reset(self):
        self.rover.reset()
        self.steps = 0
        self.scanned = np.zeros_like(self.target, dtype=bool)
        # First stationary beam footprint is scanned at reset (not rewarded).
        self.scanned |= self._laser_footprint(
            self.rover.x, self.rover.y, self.rover.x, self.rover.y
        ) & self.target
        return self.observe()

    def observe(self):
        """Compact numerical observation: pose + target/unscanned local maps.

        7 x 7 grid in the ROVER'S reference frame, spaced every 6 map units.
        Rows represent left/right, columns forward/back. Allows a PPO MLP to
        distinguish both where to drive and which parts remain unscanned.
        """
        x, y, theta = self.rover.get_state()
        ct, st = math.cos(theta), math.sin(theta)
        remaining = self.target & ~self.scanned
        if remaining.any():
            cy, cx = np.nonzero(remaining)
            dx = ((cx + 0.5).mean() - x) / self.width
            dy = ((cy + 0.5).mean() - y) / self.height
        else:
            dx, dy = 0.0, 0.0

        offsets = np.arange(-3, 4, dtype=np.float32) * 6.0
        forward, left = np.meshgrid(offsets, offsets)
        sample_x = x + forward*ct - left*st
        sample_y = y + forward*st + left*ct
        ix = np.floor(sample_x).astype(np.int32)
        iy = np.floor(sample_y).astype(np.int32)
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        local_target = np.zeros_like(sample_x, dtype=np.float32)
        local_remaining = np.zeros_like(sample_x, dtype=np.float32)
        local_target[valid] = self.target[iy[valid], ix[valid]]
        local_remaining[valid] = remaining[iy[valid], ix[valid]]

        features = np.array([
            x / self.width, y / self.height, ct, st,
            self.coverage_ratio, dx, dy,
        ], dtype=np.float32)
        return np.concatenate([
            features, local_target.ravel(), local_remaining.ravel()
        ]).astype(np.float32)

    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        x0, y0, _ = self.rover.get_state()
        valid = self.rover.move(action, bounds=(0, self.width, 0, self.height))
        x1, y1, _ = self.rover.get_state()
        new_count = 0

        if action in ("left", "right"):
            reward = -0.15       # Turning consumes time, does not scan new area.
        elif not valid:
            reward = -1.0        # Map boundary hit.
        else:
            footprint = self._laser_footprint(x0, y0, x1, y1) & self.target
            newly_scanned = footprint & ~self.scanned
            new_count = int(np.count_nonzero(newly_scanned))
            self.scanned |= newly_scanned
            if new_count > 0:
                reward = min(10.0, 10.0 * new_count / (self.beam_width * self.rover.step_size))  # Reward proportional to new area.
            elif footprint.any():
                reward = -0.5  # Only previously scanned target area.
            else:
                reward = -1.0  # Laser outside target.

        self.steps += 1
        achieved = self.coverage_ratio >= self.goal_coverage
        done = achieved or self.steps >= self.max_steps
        if achieved:
            reward += 40.0
        return self.observe(), float(reward), bool(done), {
            "coverage": self.coverage_ratio,
            "new_cells": new_count,
            "goal_reached": achieved,
            "valid_move": valid,
            "steps": self.steps,
            "time_limit_reached": self.steps >= self.max_steps and not achieved,
        }
