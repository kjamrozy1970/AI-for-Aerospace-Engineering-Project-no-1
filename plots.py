"""Matplotlib artists, independent of Qt. Create once, then update data."""
import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle


class RoverPlot:
    def __init__(self, figure):
        self.figure = figure
        self.ax = figure.add_subplot(111)
        # Preserve the target/coverage colours of the uploaded renderer.
        self.image = self.ax.imshow(
            np.zeros((100, 100), dtype=np.uint8), origin="lower",
            extent=(0, 100, 0, 100), vmin=0, vmax=2,
            cmap=ListedColormap(["white", "#d4d8de", "#75c69b"]),
            interpolation="nearest")
        self.outline, = self.ax.plot([], [], linewidth=1.0, linestyle="--",
                                     label="Target boundary", zorder=3)
        self.path, = self.ax.plot([], [], linewidth=1.5, label="Rover path")
        self.start, = self.ax.plot([], [], marker="s", linestyle="None",
                                   markersize=5, label="Start")
        self.point, = self.ax.plot([], [], marker="o", linestyle="None",
                                   markersize=7, label="Rover")
        self.beam = Circle((50, 50), radius=7.5, fill=False,
                           linestyle="--", linewidth=1.2, label="Laser footprint")
        self.ax.add_patch(self.beam)
        self.arrow = self.ax.quiver([50], [50], [6], [0],
                                    angles="xy", scale_units="xy", scale=1,
                                    width=0.008, zorder=5)
        self.ax.set_xlabel("x [map units]")
        self.ax.set_ylabel("y [map units]")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(alpha=0.15)
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        self._shape = None
        self.ax.set_title("Environment preview", fontsize=11)

    def update(self, frame):
        width, height = frame["width"], frame["height"]
        self.image.set_data(frame["image"])
        outline = frame.get("target_outline")
        if outline is not None:
            self.outline.set_data(outline[:, 0], outline[:, 1])
        else:
            self.outline.set_data([], [])
        if self._shape != (width, height):
            self.image.set_extent((0, width, 0, height))
            self.ax.set_xlim(0, width)
            self.ax.set_ylim(0, height)
            self._shape = (width, height)
        xs, ys = frame["path_x"], frame["path_y"]
        self.path.set_data(xs, ys)
        self.start.set_data(xs[:1], ys[:1])
        x, y, theta = frame["x"], frame["y"], frame["theta"]
        self.point.set_data([x], [y])
        self.beam.center = (x, y)
        self.beam.set_radius(frame["beam_width"] / 2)
        length = 0.065 * min(width, height)
        self.arrow.set_offsets(np.array([[x, y]], dtype=float))
        self.arrow.set_UVC([length * math.cos(theta)], [length * math.sin(theta)])
        label = {"train": "Training", "eval": "Policy simulation", "preview": "Preview"}[frame["mode"]]
        self.ax.set_title(
            f"{label}  |  episode {frame['episode']}  |  step {frame['steps']}/{frame['max_steps']}\n"
            f"Coverage {frame['coverage']:.1%}  ·  action: {frame['action']}", fontsize=10)


class EpisodePlot:
    def __init__(self, figure, metric="cost"):
        self.figure = figure
        self.ax = figure.add_subplot(111)
        self.metric = metric
        self.raw, = self.ax.plot([], [], linewidth=1, alpha=0.5, label="Episode")
        self.mean, = self.ax.plot([], [], linewidth=2, label="Moving average (25)")
        self.ax.set_xlabel("Completed training episode")
        self.ax.set_ylabel("Cost = −Σ rewards" if metric == "cost" else "Coverage [%]")
        self.ax.set_title("Episode cost — lower is better" if metric == "cost"
                          else "Target coverage", fontsize=10)
        self.ax.grid(alpha=0.2)
        self.ax.legend(fontsize=8)
        self.empty_text = self.ax.text(0.5, 0.5, "Waiting for a completed episode…",
                                      transform=self.ax.transAxes, ha="center", va="center", fontsize=9)
        self.reset_limits()

    def reset_limits(self):
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim((0, 105) if self.metric == "coverage" else (-1, 1))

    def update(self, records):
        x = np.array([r["episode"] for r in records], dtype=float)
        y = np.array([r[self.metric] for r in records], dtype=float)
        if self.metric == "coverage":
            y *= 100
        self.empty_text.set_visible(not len(records))
        self.raw.set_data(x, y)
        if len(records):
            window = 25
            cumulative = np.concatenate(([0.0], np.cumsum(y)))
            indices = np.arange(len(y))
            starts = np.maximum(0, indices - window + 1)
            average = (cumulative[indices + 1] - cumulative[starts]) / (indices - starts + 1)
            self.mean.set_data(x, average)
            self.ax.set_xlim(0.5, max(2.0, x[-1] + 0.5))
            if self.metric == "cost":
                lo, hi = float(y.min()), float(y.max())
                margin = max(1.0, (hi - lo) * 0.12)
                self.ax.set_ylim(lo - margin, hi + margin)
        else:
            self.mean.set_data([], [])
            self.reset_limits()


class LossPlot:
    def __init__(self, figure):
        self.figure = figure
        self.ax = figure.add_subplot(111)
        self.line, = self.ax.plot([], [], linewidth=1.6)
        self.ax.grid(alpha=0.2)
        self.ax.set_xlabel("PPO update")
        self.ax.set_title("Optimization diagnostics, not episode cost", fontsize=10)

    def update(self, records, metric="total_loss"):
        x = [record["update"] for record in records]
        y = [record[metric] for record in records]
        self.line.set_data(x, y)
        self.ax.set_ylabel(metric)
        if y:
            margin = max(0.01, (max(y) - min(y)) * 0.12)
            self.ax.set_xlim(0.5, max(2, x[-1] + 0.5))
            self.ax.set_ylim(min(y) - margin, max(y) + margin)
        else:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(-1, 1)
