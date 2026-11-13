from collections import Counter, deque
import sys
import random
import torch
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit, QLabel
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIntValidator, QDoubleValidator
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from pathlib import Path
from envinronment import Env
from agent import PPOAgent


class Demo(QMainWindow):
    def __init__(self):
        super().__init__()
        central = QWidget()
        self.setCentralWidget(central)
        # PPO variables
        self.shape = "circle"
        self.beam_width = 15
        self.step_size = 1
        self.max_steps = 100
        self.turn_angle_deg = 1
        self.goal_coverage = 1

        self.coverages = 0
        self.returns = 0
    
        self.env = Env(shape=self.shape, beam_width=self.beam_width, step_size=self.step_size, max_steps=self.max_steps, turn_angle_deg=self.turn_angle_deg, goal_coverage=self.goal_coverage)
        eval_env = Env(shape=self.shape, beam_width=self.beam_width, step_size=self.step_size, max_steps=self.max_steps, turn_angle_deg=self.turn_angle_deg, goal_coverage=self.goal_coverage)

        self.agent = PPOAgent(obs_dim=len(self.env.observe()), actions=self.env.ACTIONS)

        # Gui
        self.layout = QGridLayout(central)
        self.fig = plt.figure(figsize=(10, 10))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(321)
        self.ax3 = self.fig.add_subplot(331)

        #self.line = self.ax1.plot([], [], "b-", label = "spiral")
        #self.line = self.ax2.plot([], [], "b-", label = "spiral")

        self.layout.addWidget(self.canvas, 3, 3)

        self.button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")

        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.button_layout.addWidget(self.reset_button)

        self.vars_layout = QGridLayout()
        self.row = 1

        self.vars_title = QLabel("Variables set")
        self.vars_layout.addWidget(self.vars_title, self.row, 1)
        self.row += 1

        self.gae_label = QLabel("GAE value:")
        self.vars_layout.addWidget(self.gae_label, self.row, 0)
        self.major_radius = QLineEdit()
        self.major_radius.setText(f"Null")
        self.major_radius.setPlaceholderText("number")
        self.major_radius.setValidator(QDoubleValidator())
        self.vars_layout.addWidget(self.major_radius, self.row, 1)
        self.row += 1

        self.beam_width_label = QLabel("Beam width:")
        self.vars_layout.addWidget(self.beam_width_label, self.row, 0)
        self.beam_width_line = QLineEdit()
        self.beam_width_line.setText(f"{self.beam_width}")
        self.beam_width_line.setPlaceholderText("number")
        self.beam_width_line.setValidator(QDoubleValidator())
        self.vars_layout.addWidget(self.beam_width_line, self.row, 1)
        self.row += 1

        self.step_size_label = QLabel("Step size:")
        self.vars_layout.addWidget(self.step_size_label, self.row, 0)

        self.step_size_line = QLineEdit()
        self.step_size_line.setText(f"{self.step_size}")
        self.step_size_line.setPlaceholderText("int")
        self.step_size_line.setValidator(QIntValidator())
        self.vars_layout.addWidget(self.step_size_line, self.row, 1)
        self.row += 1

        self.max_steps_label = QLabel("Max steps:")
        self.vars_layout.addWidget(self.max_steps_label, self.row, 0)

        self.max_steps_line = QLineEdit()
        self.max_steps_line.setText(f"{self.max_steps}")
        self.max_steps_line.setPlaceholderText("int")
        self.max_steps_line.setValidator(QIntValidator())
        self.vars_layout.addWidget(self.max_steps_line, self.row, 1)
        self.row += 1

        self.turn_angle_deg_label = QLabel("Turn angle [deg]:")
        self.vars_layout.addWidget(self.turn_angle_deg_label, self.row, 0)

        self.turn_angle_deg_line = QLineEdit()
        self.turn_angle_deg_line.setText(f"{self.turn_angle_deg}")
        self.turn_angle_deg_line.setPlaceholderText("int")
        self.turn_angle_deg_line.setValidator(QIntValidator())
        self.vars_layout.addWidget(self.max_steps_line, self.row, 1)
        self.row += 1

        self.goal_coverage_label = QLabel("Goal covarage:")
        self.vars_layout.addWidget(self.goal_coverage_label, self.row, 0)

        self.goal_coverage_line = QLineEdit()
        self.goal_coverage_line.setText(f"{self.goal_coverage}")
        self.goal_coverage_line.setPlaceholderText("int")
        self.goal_coverage_line.setValidator(QIntValidator())
        self.vars_layout.addWidget(self.goal_coverage_line, self.row, 1)
        self.row += 1

        #self.shape_ circle triangle

        self.layout.addLayout(self.button_layout, 4, 3)
        self.layout.addLayout(self.vars_layout, 4, 2)


        # Timer
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_animation)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.timer.stop)
        self.reset_button.clicked.connect(self.reset_animation)

    def start(self):
        self.coverages, self.returns = self.train(self.agent, self.env, updates=40, rollout_steps=512, log_interval=5)
        self.timer.start()

    def update_animation(self):
       self.plot_results(self.env, self.returns, self.coverages, "./result/", "result.png")

    def reset_animation(self):
        pass

    def run_test(self, agent, env, deterministic=True):
        obs = env.reset()
        total_reward = 0.0
        actions = Counter()
        while True:
            idx, _, _ = agent.choose_action(obs, deterministic=deterministic)
            name = agent.actions[idx]
            obs, reward, done, info = env.step(name)
            total_reward += reward
            actions[name] += 1
            if done:
                break
        return {"coverage": env.coverage_ratio, "reward": total_reward,
                "actions": dict(actions), "goal_reached": info["goal_reached"]}


    def plot_results(self, env, returns, coverages, output_path, title):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
        # Background target / completed area and the actual movement path.
        image = np.zeros((env.height, env.width), dtype=np.uint8)
        image[env.target] = 1
        image[env.target & env.scanned] = 2
        from matplotlib.colors import ListedColormap
        self.ax1.imshow(image, origin="lower", extent=(0, env.width, 0, env.height), cmap=ListedColormap(["white", "#d4d8de", "#75c69b"]), vmin=0, vmax=2, interpolation="nearest")
        x, y = np.array(env.rover.x_list), np.array(env.rover.y_list)
        self.ax1.plot(x, y, color="#111827", lw=1.3, alpha=0.85, label="Rover path")
        # Arrows show heading, including turns in place which have no x-y displacement.
        if len(x) > 1:
            indices = np.unique(np.linspace(0, len(x)-1, min(15, len(x)), dtype=int))
            headings = np.array(env.rover.theta_list)[indices]
            self.ax1.quiver(x[indices], y[indices], 5*np.cos(headings), 5*np.sin(headings), color="#be123c", angles="xy", scale_units="xy", scale=1, width=0.004, label="Heading")
        self.ax1.scatter(x[0], y[0], color="#1d4ed8", s=60, marker="s", label="Start", zorder=4)
        self.ax1.scatter(x[-1], y[-1], color="#f97316", s=60, marker="o", label="End", zorder=4)
        self.ax1.set_xlim(0, env.width)
        self.ax1.set_ylim(0, env.height)
        self.ax1.set_aspect("equal", adjustable="box")
        self.ax1.set_title(f"{title}\nCoverage: {env.coverage_ratio:.1%}")
        self.ax1.set_xlabel("x [map units]")
        self.ax1.set_ylabel("y [map units]")
        self.ax1.legend(loc="lower right", fontsize=8)
        if len(coverages):
            self.ax2.plot(np.arange(len(coverages))+1, np.array(coverages)*100, color="#16a34a", label="Episode coverage (%)")
        else:
            self.ax2.text(0.5, 0.5, "Evaluation only: no training history", transform=ax2.transAxes, ha="center", va="center")
        self.ax2.set_xlabel("Training episode")
        self.ax2.set_ylabel("Coverage (%)")
        self.ax2.set_ylim(0, 105)
        self.ax2.grid(alpha=0.2)
        if len(coverages):
            self.ax2.legend(loc="upper left")
        self.ax3 = self.ax2.twinx()
        if len(returns):
            self.ax3.plot(np.arange(len(returns))+1, returns, color="#2563eb", alpha=0.35, label="Episode reward")
        self.ax3.set_ylabel("Episode reward")
        self.fig.tight_layout()
        self.fig.savefig(output_path, dpi=160)
        plt.close(self.fig)
        self.canvas.draw_idle()


    def train(self, agent, env, updates=40, rollout_steps=512, log_interval=5):
        observations = env.reset()
        episode_reward = 0.0
        recent = deque(maxlen=25)
        recent_rewards = deque(maxlen=25)
        episode_returns, episode_coverages = [], []
        episode_num = 0
        for update in range(1, updates+1):
            memory = []
            for _ in range(rollout_steps):
                action_idx, logprob, value = agent.choose_action(observations)
                observations_next, reward, done, info = env.step(agent.actions[action_idx])
                memory.append((observations, action_idx, logprob, value, reward / 10.0, done))
                observations = observations_next
                episode_reward += reward
                if done:
                    episode_returns.append(episode_reward)
                    episode_coverages.append(info["coverage"])
                    recent.append(info["coverage"])
                    recent_rewards.append(episode_reward)
                    episode_num += 1
                    observations = env.reset()
                    episode_reward = 0.0

            # Bootstrap the last unfinished episode in this rollout.
            _, _, last_value = agent.choose_action(observations, deterministic=True)
            agent.update(memory, bootstrap_value=last_value)
            if update == 1 or update % log_interval == 0 or update == updates:
                recent_mean = np.mean(recent) if recent else float("nan")
                recent_reward = np.mean(recent_rewards) if recent_rewards else float("nan")
                print(f"PPO update={update:3d}/{updates} episodes={episode_num:4d} "
                      f"mean_coverage_25={recent_mean:.1%} "
                      f"mean_return_25={recent_reward:.1f}")
        return episode_returns, episode_coverages
