"""Training/evaluation without Qt, reusable in tests and in the GUI worker.

The PPO objective, reward scaling, and finite-episode/time-limit semantics
are retained from the uploaded project. A training stop discards a partial
rollout, but preserves the latest completed optimizer update.
"""
from collections import Counter
import copy
import random
import threading
import time

import numpy as np
import torch

from agent import PPOAgent
from config import Config
from envinronment import Env


class RunControl:
    """Only synchronization object shared by the GUI and the worker."""
    def __init__(self, delay_ms=0):
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._delay_ms = max(0, int(delay_ms))

    def stop(self):
        self.stop_event.set()

    def set_delay(self, delay_ms):
        with self._lock:
            self._delay_ms = max(0, int(delay_ms))

    def wait_step(self):
        with self._lock:
            seconds = self._delay_ms / 1000.0
        return self.stop_event.wait(seconds) if seconds else self.stop_event.is_set()


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def frame_snapshot(env, *, mode="preview", episode=0, reward=0.0, action="—"):
    """Create an independent numerical snapshot; never pass a live env to Qt."""
    image = np.zeros((env.height, env.width), dtype=np.uint8)
    image[env.target] = 1
    image[env.target & env.scanned] = 2
    return {
        "image": image,
        "shape": env.shape,
        "target_outline": env.outline.copy(),
        "path_x": np.array(env.rover.x_list, dtype=float, copy=True),
        "path_y": np.array(env.rover.y_list, dtype=float, copy=True),
        "x": float(env.rover.x), "y": float(env.rover.y),
        "theta": float(env.rover.theta), "width": env.width, "height": env.height,
        "beam_width": env.beam_width, "coverage": env.coverage_ratio,
        "steps": env.steps, "max_steps": env.max_steps,
        "episode": int(episode), "reward": float(reward),
        "action": action, "mode": mode,
    }


def make_preview(config):
    return frame_snapshot(Env(**config.env_kwargs()))


class _FramePublisher:
    def __init__(self, callback):
        self.callback = callback
        self.last_time = -float("inf")

    def send(self, env, *, force=False, **kwargs):
        now = time.monotonic()
        if self.callback is not None and (force or now - self.last_time >= 0.1):
            self.last_time = now
            self.callback(frame_snapshot(env, **kwargs))


def _make_agent(config, env, checkpoint, *, restore_optimizer):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch installation. Select CPU.")
    agent = PPOAgent(obs_dim=len(env.observe()), actions=env.ACTIONS,
                     **config.agent_kwargs())
    if checkpoint is not None:
        if tuple(checkpoint["actions"]) != tuple(agent.actions):
            raise ValueError("The saved model uses a different action list.")
        if checkpoint.get("obs_dim", len(env.observe())) != len(env.observe()):
            raise ValueError("The saved model uses a different observation dimension.")
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        if restore_optimizer and checkpoint.get("optimizer_state_dict"):
            # Loading may move optimizer tensors to a different device; copy
            # first so the caller's CPU checkpoint remains independent.
            agent.optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))
    return agent


def train_session(config: Config, checkpoint=None, *, control=None,
                  on_frame=None, on_episode=None, on_update=None, on_status=None):
    """Run `config.updates` *additional* PPO updates, or stop cooperatively.

    Callback data contains copies/scalars only. Rewards shown in the GUI are
    unscaled episode sums; PPO receives rewards multiplied by reward_scale.
    """
    config.validate()
    control = control or RunControl()
    old_updates = len(checkpoint.get("update_history", [])) if checkpoint else 0
    random.seed(config.seed + old_updates)
    np.random.seed((config.seed + old_updates) % (2**32))
    torch.manual_seed(config.seed + old_updates)
    env = Env(**config.env_kwargs())
    if env.coverage_ratio >= config.goal_coverage:
        raise ValueError("The laser already meets the coverage goal at the start position. "
                         "Reduce the beam or enlarge the target before training.")
    agent = _make_agent(config, env, checkpoint, restore_optimizer=True)
    agent.model.train()
    episode_history = copy.deepcopy(checkpoint.get("episode_history", [])) if checkpoint else []
    update_history = copy.deepcopy(checkpoint.get("update_history", [])) if checkpoint else []
    obs = env.observe()
    episode_reward = 0.0
    path_length = 0.0
    counts = Counter()
    publisher = _FramePublisher(on_frame)
    episode_num = len(episode_history) + 1
    publisher.send(env, force=True, mode="train", episode=episode_num)
    last_action = "—"
    last_completed_frame = None
    if on_status:
        on_status("PPO training: collecting environment transitions.")

    for update_index in range(config.updates):
        memory = []
        for _ in range(config.rollout_steps):
            if control.stop_event.is_set():
                break
            x0, y0, _ = env.rover.get_state()
            idx, logprob, value = agent.choose_action(obs)
            last_action = agent.actions[idx]
            obs_next, reward, done, info = env.step(last_action)
            memory.append((obs, idx, logprob, value, reward * config.reward_scale, done))
            obs = obs_next
            episode_reward += reward
            counts[last_action] += 1
            path_length += float(np.hypot(env.rover.x - x0, env.rover.y - y0))
            publisher.send(env, mode="train", episode=episode_num,
                           reward=episode_reward, action=last_action)
            if done:
                last_completed_frame = frame_snapshot(env, mode="train", episode=episode_num,
                                                      reward=episode_reward, action=last_action)
                record = {
                    "episode": episode_num, "reward": float(episode_reward),
                    "cost": float(-episode_reward), "coverage": float(info["coverage"]),
                    "steps": int(env.steps), "path_length": float(path_length),
                    "left": int(counts["left"]), "forward": int(counts["forward"]),
                    "right": int(counts["right"]), "goal_reached": bool(info["goal_reached"]),
                }
                episode_history.append(record)
                if on_episode:
                    on_episode(dict(record))
                obs = env.reset()
                episode_num += 1
                episode_reward = 0.0
                path_length = 0.0
                counts.clear()
            if control.wait_step():
                break

        if control.stop_event.is_set():
            # Do not train on an incomplete rollout after the user presses Stop.
            break
        bootstrap = agent.value(obs)
        if on_status:
            on_status(f"PPO update {old_updates + update_index + 1}…")
        agent.update(memory, bootstrap_value=bootstrap)
        record = {"update": old_updates + update_index + 1,
                  "episodes": len(episode_history), **agent.last_metrics}
        update_history.append(record)
        if on_update:
            on_update(dict(record))

    if env.steps == 0 and last_completed_frame is not None and on_frame is not None:
        on_frame(last_completed_frame)
    else:
        publisher.send(env, force=True, mode="train", episode=episode_num,
                       reward=episode_reward, action=last_action)
    result = {
        "format_version": 2,
        "config": config.to_dict(),
        "obs_dim": int(len(env.observe())),
        "actions": agent.actions,
        "model_state_dict": _cpu_copy(agent.model.state_dict()),
        "optimizer_state_dict": _cpu_copy(agent.optimizer.state_dict()),
        "episode_history": episode_history,
        "update_history": update_history,
    }
    return {"checkpoint": result, "stopped": control.stop_event.is_set(),
            "partial_episode_steps": env.steps}


def evaluate_session(config: Config, checkpoint, *, deterministic=True,
                     control=None, on_frame=None, on_status=None):
    """One separate evaluation episode. It never updates weights or training logs."""
    if checkpoint is None:
        raise ValueError("Train or load a model first.")
    control = control or RunControl(delay_ms=40)
    config.validate()
    env = Env(**config.env_kwargs())
    agent = _make_agent(config, env, checkpoint, restore_optimizer=False)
    agent.model.eval()
    obs = env.observe()
    publisher = _FramePublisher(on_frame)
    total_reward = 0.0
    counts = Counter()
    goal = env.coverage_ratio >= env.goal_coverage
    path_length = 0.0
    action = "—"
    publisher.send(env, force=True, mode="eval", episode=1)
    if on_status:
        on_status("Policy simulation — no model updates.")
    while not goal and env.steps < env.max_steps and not control.stop_event.is_set():
        x0, y0, _ = env.rover.get_state()
        idx, _, _ = agent.choose_action(obs, deterministic=deterministic)
        action = agent.actions[idx]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        counts[action] += 1
        path_length += float(np.hypot(env.rover.x - x0, env.rover.y - y0))
        goal = bool(info["goal_reached"])
        publisher.send(env, mode="eval", episode=1, reward=total_reward, action=action)
        if done or control.wait_step():
            break
    publisher.send(env, force=True, mode="eval", episode=1,
                   reward=total_reward, action=action)
    return {"stopped": control.stop_event.is_set(), "evaluation": {
        "reward": float(total_reward), "cost": float(-total_reward),
        "coverage": env.coverage_ratio, "steps": env.steps,
        "path_length": path_length, "actions": dict(counts), "goal_reached": goal,
    }}
