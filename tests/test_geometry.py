"""Geometry, configuration migration and training integration regression tests."""
from dataclasses import replace
import math
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from config import Config
from engine import make_preview, frame_snapshot, train_session, evaluate_session, RunControl
from envinronment import Env
from geometry import SHAPES, validate_shape_parameters


def test_circle_uses_explicit_center_and_radius():
    config = Config(circle_cx=30, circle_cy=45, circle_radius=12,
                    start_x=5, start_y=8)
    env = Env(**config.env_kwargs())
    expected = (env.X - 30)**2 + (env.Y - 45)**2 <= 12**2
    assert np.array_equal(env.target, expected)
    assert env.center == (30, 45) and env.radius == 12
    assert not env.target[50, 50]
    assert env.rover.get_state() == (5, 8, 0)


def test_ellipse_axes_are_radii_not_diameters():
    env = Env(**Config(shape="ellipse", ellipse_cx=40, ellipse_cy=60,
                       ellipse_rx=20, ellipse_ry=10).env_kwargs())
    expected = ((env.X-40)/20)**2 + ((env.Y-60)/10)**2 <= 1
    assert np.array_equal(env.target, expected)
    assert env.target[60, 59] and not env.target[60, 60]
    assert env.target[69, 40] and not env.target[70, 40]
    assert abs(env.target_count - math.pi*20*10) / (math.pi*20*10) < 0.03


def test_ellipse_can_have_vertical_radius_larger_than_horizontal():
    env = Env(**Config(shape="ellipse", ellipse_rx=10, ellipse_ry=35).env_kwargs())
    assert np.isclose(env.outline[:, 0].ptp() if hasattr(env.outline[:, 0], 'ptp')
                      else np.ptp(env.outline[:, 0]), 20)
    assert np.isclose(np.ptp(env.outline[:, 1]), 70)


def test_equal_ellipse_radii_match_circle():
    circle = Env(**Config(circle_radius=20).env_kwargs())
    ellipse = Env(**Config(shape="ellipse", ellipse_rx=20, ellipse_ry=20).env_kwargs())
    assert np.array_equal(circle.target, ellipse.target)


@pytest.mark.parametrize("points", [((10, 10), (40, 10), (10, 40)),
                                     ((10, 40), (40, 10), (10, 10))])
def test_triangle_order_and_boundary(points):
    params = {"vertices": points}
    env = Env(shape="triangle", shape_params=params)
    expected = (env.X >= 10) & (env.Y >= 10) & (env.X + env.Y <= 50)
    assert np.array_equal(env.target, expected)
    assert env.target[29, 20]  # (20.5, 29.5) lies on the hypotenuse
    assert not env.target[30, 20]


@pytest.mark.parametrize("p1,p2", [((12, 18), (72, 48)), ((72, 48), (12, 18)),
                                  ((12, 48), (72, 18))])
def test_rectangle_accepts_opposite_corners_in_any_order(p1, p2):
    env = Env(shape="rectangle", shape_params={"point1": p1, "point2": p2})
    assert env.target_count == 60 * 30
    assert env.target[18, 12] and env.target[47, 71]
    assert not env.target[48, 71]


def test_square_matches_equal_sided_rectangle():
    square = Env(**Config(shape="square").env_kwargs())
    rectangle = Env(**Config(shape="rectangle").env_kwargs())
    assert np.array_equal(square.target, rectangle.target)
    assert square.target_count == 3600


def test_square_rejects_unequal_sides():
    with pytest.raises(ValueError, match="For a square"):
        Config(shape="square", rect_x2=70, rect_y2=80).validate()


@pytest.mark.parametrize("changes", [
    {"circle_radius": 0},
    {"circle_cx": 10, "circle_radius": 11},
    {"shape": "ellipse", "ellipse_rx": 0},
    {"shape": "ellipse", "ellipse_ry": 80},
    {"shape": "triangle", "triangle_x3": 50, "triangle_y3": 20},
    {"shape": "triangle", "triangle_x3": -1},
    {"shape": "rectangle", "rect_x2": 20},
    {"shape": "rectangle", "rect_y1": -1},
    {"start_x": 101}, {"start_y": -1}, {"start_x": float("nan")},
    {"ellipse_rx": float("inf")},
])
def test_invalid_geometry_is_rejected_without_clipping(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()


def test_empty_raster_target_has_readable_error():
    with pytest.raises(ValueError, match="cell centres"):
        Env(**Config(circle_cx=20.0, circle_cy=20.0, circle_radius=0.01).env_kwargs())


def test_inactive_shapes_may_be_outside_smaller_map():
    cfg = Config(width=40, height=40, circle_cx=20, circle_cy=20,
                 circle_radius=10, start_x=5, start_y=6)
    env = Env(**cfg.env_kwargs())
    assert env.target.shape == (40, 40)


def test_start_position_heading_and_reset_use_exact_point():
    cfg = Config(start_x=12.25, start_y=15.75, start_theta_deg=90)
    env = Env(**cfg.env_kwargs())
    assert env.rover.get_state() == pytest.approx((12.25, 15.75, math.pi/2))
    env.step("forward")
    assert env.rover.get_state() == pytest.approx((12.25, 18.75, math.pi/2))
    env.step("left")
    env.reset()
    assert env.rover.x_list == [12.25] and env.rover.y_list == [15.75]
    assert env.rover.get_state() == pytest.approx((12.25, 15.75, math.pi/2))
    assert not env.scanned.any()  # outside target is a permitted starting position


@pytest.mark.parametrize("shape", SHAPES)
def test_all_shapes_use_same_observation_and_respect_swept_laser(shape):
    cfg = Config(shape=shape, beam_width=4, step_size=20)
    env = Env(**cfg.env_kwargs())
    assert env.observe().shape == (105,)
    before = env.scanned.copy()
    env.step("forward")
    assert not before[50, 60] and env.scanned[50, 60]
    assert np.isfinite(env.observe()).all()
    assert np.all(~env.scanned | env.target)
    assert np.allclose(env.outline[0], env.outline[-1])


@pytest.mark.parametrize("shape", SHAPES)
def test_geometry_survives_checkpoint_and_policy_evaluation(shape, tmp_path):
    cfg = Config(shape=shape, updates=1, rollout_steps=32, minibatch_size=16,
                 update_epochs=1, max_steps=8, start_x=40, start_y=45)
    records = []
    cp = train_session(cfg, on_episode=records.append)["checkpoint"]
    assert len(records) == 4
    assert len(cp["update_history"]) == 1
    assert np.isfinite(cp["update_history"][0]["total_loss"])
    path = tmp_path / "model.pt"
    torch.save(cp, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    restored = Config.from_dict(loaded["config"])
    assert restored == cfg
    assert np.array_equal(make_preview(cfg)["image"], make_preview(restored)["image"])
    result = evaluate_session(restored, loaded, control=RunControl())
    assert 0 <= result["evaluation"]["coverage"] <= 1
    assert result["evaluation"]["steps"] == 8


def test_config_json_roundtrip():
    cfg = Config(shape="ellipse", ellipse_cx=35.25, ellipse_rx=12.125,
                 ellipse_ry=7.5, start_x=5.5, start_y=8.75)
    assert Config.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


@pytest.mark.parametrize("shape", ("circle", "triangle"))
@pytest.mark.parametrize("center_start", (True, False))
def test_legacy_gui_config_migrates_without_changing_geometry(shape, center_start):
    data = dict(width=80, height=60, target_scale=1.2, shape=shape,
                start_center=center_start, start_x=10, start_y=15, start_theta_deg=45)
    cfg = Config.from_dict(data)
    start = (40, 30, math.pi/4) if center_start else (10, 15, math.pi/4)
    old_api = Env(width=80, height=60, shape=shape, target_scale=1.2, start=start)
    explicit = Env(**cfg.env_kwargs())
    assert np.array_equal(old_api.target, explicit.target)
    assert np.array_equal(old_api.observe(), explicit.observe())
    assert explicit.rover.get_state() == pytest.approx(start)
    assert "target_scale" not in cfg.to_dict() and "start_center" not in cfg.to_dict()


def test_mutating_snapshot_does_not_change_geometry():
    env = Env(shape="ellipse")
    frame = frame_snapshot(env)
    saved = env.outline.copy()
    frame["target_outline"][:] = -100
    assert np.array_equal(env.outline, saved)


def test_active_source_has_no_polish_ui_text():
    # Archived originals are intentionally untouched for regression comparison.
    root = Path(__file__).resolve().parents[1]
    polish_letters = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
    for name in ("window.py", "plots.py", "engine.py", "config.py", "geometry.py"):
        text = (root / name).read_text()
        assert not polish_letters.intersection(text), name
