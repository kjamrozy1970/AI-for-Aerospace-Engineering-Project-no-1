import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from agent import PPOAgent
from config import Config
from engine import RunControl, evaluate_session, frame_snapshot, train_session
from envinronment import Env
from new_rover import Rover
from plots import RoverPlot, EpisodePlot, LossPlot


@pytest.fixture
def small_config():
    return Config(updates=2, rollout_steps=64, minibatch_size=32,
                  update_epochs=2, max_steps=16)


@pytest.fixture
def trained(small_config):
    return train_session(small_config)["checkpoint"]


def test_left_and_right_are_rotations():
    rover = Rover(10, 10, 0, step_size=3, turn_angle_deg=15)
    rover.move("left")
    assert rover.x == 10 and rover.y == 10
    assert rover.theta == pytest.approx(math.radians(15))
    rover.move("right")
    assert rover.theta == pytest.approx(0)


def test_forward_uses_heading():
    rover = Rover(10, 10, math.pi / 2, step_size=3)
    rover.move("forward")
    assert rover.x == pytest.approx(10)
    assert rover.y == pytest.approx(13)


def test_reset_restores_pose_and_clears_history():
    rover = Rover(12, 13, math.pi)
    rover.move("forward")
    rover.move("left")
    rover.reset()
    assert rover.get_state() == (12, 13, math.pi)
    assert rover.x_list == [12] and rover.action_history == []
    assert len(rover.theta_list) == 1


def test_boundary_rejects_forward():
    rover = Rover(99, 50, 0, step_size=3)
    assert not rover.move("forward", bounds=(0, 100, 0, 100))
    assert rover.x == 99


@pytest.mark.parametrize("shape", ["circle", "triangle"])
def test_masks_and_observation(shape):
    env = Env(shape=shape)
    assert env.target_count > 0
    assert env.observe().shape == (105,)
    assert env.observe().dtype == np.float32
    assert 0 < env.coverage_ratio < 1
    assert np.isfinite(env.observe()).all()


def test_turn_cannot_scan_new_area():
    env = Env()
    previous = env.scanned.copy()
    _, reward, done, info = env.step("left")
    assert np.array_equal(env.scanned, previous)
    assert reward == pytest.approx(-0.15)
    assert info["new_cells"] == 0 and not done


def test_whole_swept_beam_is_scanned():
    env = Env(beam_width=4, step_size=20)
    before = env.scanned.copy()
    env.step("forward")
    # The middle of a long move is included, not only the final beam disk.
    assert not before[50, 60]
    assert env.scanned[50, 60]


def test_target_scales_with_map_dimensions():
    env = Env(width=80, height=60, start=(40, 30, 0), target_scale=1.2)
    assert env.center == (40, 30)
    assert env.radius == pytest.approx(18)
    assert env.target.shape == (60, 80)


@pytest.mark.parametrize("shape", ["circle", "triangle"])
def test_default_environment_matches_original(shape):
    path = Path(__file__).resolve().parents[1] / "original_sources" / "envinronment.py"
    spec = importlib.util.spec_from_file_location("original_environment", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    old, new = original.Env(shape=shape), Env(shape=shape)
    assert np.array_equal(old.target, new.target)
    for action in ["left", "forward", "forward", "right", "forward"] * 5:
        o1, r1, d1, i1 = old.step(action)
        o2, r2, d2, i2 = new.step(action)
        assert np.array_equal(o1, o2)
        assert r1 == r2 and d1 == d2 and i1 == i2


def test_snapshot_has_no_shared_arrays():
    env = Env()
    snap = frame_snapshot(env)
    saved = snap["image"].copy()
    env.step("forward")
    assert np.array_equal(saved, snap["image"])
    assert len(snap["path_x"]) == 1


@pytest.mark.parametrize("kwargs", [dict(lr=0), dict(beam_width=-1),
    dict(minibatch_size=1024, rollout_steps=64), dict(turn_angle_deg=0),
    dict(start_x=200), dict(shape="unknown")])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs).validate()


def test_train_logs_cost_and_updates_weights(small_config):
    torch.manual_seed(small_config.seed)
    initial = PPOAgent(105, **small_config.agent_kwargs())
    before = {k: v.clone() for k, v in initial.model.state_dict().items()}
    episodes, updates, frames = [], [], []
    cp = train_session(small_config, on_episode=episodes.append,
                       on_update=updates.append, on_frame=frames.append)["checkpoint"]
    assert len(updates) == 2
    assert len(episodes) == 8
    assert frames[-1]["steps"] == 16
    assert all(r["cost"] == -r["reward"] for r in episodes)
    assert all(r["left"] + r["right"] + r["forward"] == r["steps"] for r in episodes)
    assert all(np.isfinite(r["total_loss"]) for r in updates)
    assert any(not torch.equal(before[k], cp["model_state_dict"][k]) for k in before)


def test_continue_keeps_history_and_adds_updates(trained, small_config):
    cp = train_session(small_config, trained)["checkpoint"]
    assert len(cp["update_history"]) == 4
    assert cp["update_history"][-1]["update"] == 4
    assert len(cp["episode_history"]) == 16
    assert cp["episode_history"][:8] == trained["episode_history"]


def test_stop_does_not_apply_partial_rollout(trained, small_config):
    control = RunControl()
    def stop_on_first_frame(_):
        control.stop()
    result = train_session(small_config, trained, control=control, on_frame=stop_on_first_frame)
    assert result["stopped"]
    cp = result["checkpoint"]
    assert len(cp["update_history"]) == len(trained["update_history"])
    for name, tensor in trained["model_state_dict"].items():
        assert torch.equal(tensor, cp["model_state_dict"][name])


def test_evaluation_does_not_change_model_or_logs(trained, small_config):
    saved = copy.deepcopy(trained)
    result = evaluate_session(small_config, trained, control=RunControl())
    assert result["evaluation"]["steps"] == 16
    assert 0 <= result["evaluation"]["coverage"] <= 1
    assert trained["episode_history"] == saved["episode_history"]
    for name, tensor in saved["model_state_dict"].items():
        assert torch.equal(tensor, trained["model_state_dict"][name])


def test_checkpoint_roundtrip_weights_only(trained, tmp_path):
    path = tmp_path / "model.pt"
    torch.save(trained, path)
    cp = torch.load(path, weights_only=True, map_location="cpu")
    assert cp["config"] == trained["config"]
    assert cp["episode_history"] == trained["episode_history"]


def test_metrics_only_extension_preserves_original_ppo_update():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("original_agent", root / "original_sources" / "agent.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(12)
    old = original.PPOAgent(105, update_epochs=2, minibatch_size=8)
    new = PPOAgent(105, update_epochs=2, minibatch_size=8)
    new.model.load_state_dict(old.model.state_dict())
    env = Env(max_steps=16)
    buffer = []
    obs = env.observe()
    for _ in range(16):
        action, logp, value = old.choose_action(obs)
        nxt, reward, done, _ = env.step(old.actions[action])
        buffer.append((obs, action, logp, value, reward / 10, done))
        obs = nxt
    torch.manual_seed(20)
    old.update(buffer)
    torch.manual_seed(20)
    new.update(buffer)
    for key, tensor in old.model.state_dict().items():
        assert torch.equal(tensor, new.model.state_dict()[key])
    assert "total_loss" in new.last_metrics


def test_renderers_update_existing_artists_without_new_axes(trained):
    env = Env()
    env.step("forward")
    plot_types = [RoverPlot, lambda f: EpisodePlot(f, "cost"),
                  lambda f: EpisodePlot(f, "coverage"), LossPlot]
    for make_plot in plot_types:
        figure = Figure(figsize=(5, 4), layout="constrained")
        canvas = FigureCanvasAgg(figure)
        plot = make_plot(figure)
        before = len(plot.ax.lines), len(plot.ax.patches), len(plot.ax.collections)
        for _ in range(3):
            if isinstance(plot, RoverPlot):
                plot.update(frame_snapshot(env))
            elif isinstance(plot, EpisodePlot):
                plot.update(trained["episode_history"])
            else:
                plot.update(trained["update_history"])
            canvas.draw()
        assert len(figure.axes) == 1
        assert before == (len(plot.ax.lines), len(plot.ax.patches), len(plot.ax.collections))
