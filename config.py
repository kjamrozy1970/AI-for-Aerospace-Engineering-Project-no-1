"""Validated configuration with explicit geometry in map coordinates."""
from dataclasses import asdict, dataclass
import math
from geometry import SHAPES, default_shape_parameters, validate_shape_parameters


@dataclass(frozen=True)
class Config:
    width: int = 100
    height: int = 100
    shape: str = "circle"
    circle_cx: float = 50.0
    circle_cy: float = 50.0
    circle_radius: float = 25.0
    ellipse_cx: float = 50.0
    ellipse_cy: float = 50.0
    ellipse_rx: float = 30.0
    ellipse_ry: float = 20.0
    triangle_x1: float = 20.0
    triangle_y1: float = 20.0
    triangle_x2: float = 80.0
    triangle_y2: float = 20.0
    triangle_x3: float = 50.0
    triangle_y3: float = 80.0
    rect_x1: float = 20.0
    rect_y1: float = 20.0
    rect_x2: float = 80.0
    rect_y2: float = 80.0
    beam_width: float = 15.0
    step_size: float = 3.0
    turn_angle_deg: float = 15.0
    max_steps: int = 180
    goal_coverage: float = 0.98
    start_x: float = 50.0
    start_y: float = 50.0
    start_theta_deg: float = 0.0
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    updates: int = 40
    rollout_steps: int = 512
    reward_scale: float = 0.1
    seed: int = 2026
    device: str = "cpu"

    def shape_parameters(self):
        if self.shape == "circle":
            return dict(center=[self.circle_cx, self.circle_cy], radius=self.circle_radius)
        if self.shape == "ellipse":
            return dict(center=[self.ellipse_cx, self.ellipse_cy],
                        radii=[self.ellipse_rx, self.ellipse_ry])
        if self.shape == "triangle":
            return dict(vertices=[[self.triangle_x1, self.triangle_y1],
                                  [self.triangle_x2, self.triangle_y2],
                                  [self.triangle_x3, self.triangle_y3]])
        if self.shape in ("rectangle", "square"):
            return dict(point1=[self.rect_x1, self.rect_y1],
                        point2=[self.rect_x2, self.rect_y2])
        raise ValueError(f"Unsupported target shape: {self.shape!r}")

    def validate(self):
        for name, value in asdict(self).items():
            if name in ("shape", "device"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        for name in ("width", "height", "max_steps", "update_epochs",
                     "minibatch_size", "updates", "rollout_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not 16 <= self.width <= 500 or not 16 <= self.height <= 500:
            raise ValueError("Map width and height must be between 16 and 500.")
        if self.shape not in SHAPES:
            raise ValueError("Available shapes: circle, ellipse, triangle, rectangle, square.")
        validate_shape_parameters(self.shape, self.shape_parameters(), self.width, self.height)
        if self.beam_width <= 0 or self.step_size <= 0:
            raise ValueError("Laser diameter and forward step must be positive.")
        if not 0 < self.turn_angle_deg <= 180:
            raise ValueError("Turn angle must be in (0, 180] degrees.")
        if not 0 < self.goal_coverage <= 1:
            raise ValueError("Target coverage must be in (0, 1].")
        if not (0 <= self.start_x <= self.width and 0 <= self.start_y <= self.height):
            raise ValueError("Rover start (x, y) must be inside the map. "
                             "It does not need to be inside the target shape.")
        if not 0 < self.lr <= 0.1:
            raise ValueError("Learning rate must be in (0, 0.1].")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Gamma and GAE lambda must be in [0, 1].")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("PPO clip must be in (0, 1).")
        if self.entropy_coef < 0 or self.value_coef < 0 or self.reward_scale <= 0:
            raise ValueError("Invalid PPO coefficients or reward scale.")
        if self.rollout_steps < 8 or not 2 <= self.minibatch_size <= self.rollout_steps:
            raise ValueError("Required: rollout >= 8 and 2 <= minibatch <= rollout.")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("Seed must be an integer from 0 to 2147483647.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Device must be cpu or cuda.")
        return self

    def env_kwargs(self):
        self.validate()
        return dict(width=self.width, height=self.height, shape=self.shape,
                    beam_width=self.beam_width, step_size=self.step_size,
                    turn_angle_deg=self.turn_angle_deg, max_steps=self.max_steps,
                    goal_coverage=self.goal_coverage,
                    start=(self.start_x, self.start_y, math.radians(self.start_theta_deg)),
                    shape_params=self.shape_parameters())

    def agent_kwargs(self):
        return {name: getattr(self, name) for name in (
            "lr", "gamma", "gae_lambda", "clip_epsilon", "entropy_coef",
            "value_coef", "update_epochs", "minibatch_size", "device")}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Read new checkpoints and migrate the previous GUI's scale/centre fields."""
        values = dict(data)
        legacy = "target_scale" in values or "start_center" in values
        scale = values.pop("target_scale", 1.0)
        center_start = values.pop("start_center", False)
        if legacy:
            width, height = values.get("width", 100), values.get("height", 100)
            shape = values.get("shape", "circle")
            if shape == "circle" and "circle_radius" not in values:
                p = default_shape_parameters(shape, width, height, scale)
                values.update(circle_cx=p["center"][0], circle_cy=p["center"][1],
                              circle_radius=p["radius"])
            elif shape == "triangle" and "triangle_x1" not in values:
                p = default_shape_parameters(shape, width, height, scale)
                for i, (x, y) in enumerate(p["vertices"], 1):
                    values[f"triangle_x{i}"] = x
                    values[f"triangle_y{i}"] = y
            if center_start:
                values["start_x"], values["start_y"] = width / 2, height / 2
        return cls(**values).validate()
