from __future__ import annotations

import csv
from pathlib import Path

import torch
from PySide6.QtCore import Qt, QTimer, Slot, QLocale
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QVBoxLayout, QWidget)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from config import Config
from engine import make_preview
from plots import RoverPlot, EpisodePlot, LossPlot
from worker import TrainingWorker


class Demo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        self.setWindowTitle("PPO Rover Lab — Training and Simulation")
        self.resize(1480, 930)
        self.setMinimumSize(1050, 720)
        self.worker = None
        self.checkpoint = None
        self.active_config = None
        self.episode_history = []
        self.update_history = []
        self._pending_frame = None
        self._last_frame = None
        self._charts_dirty = False
        self._pending_reset = None
        self._closing = False
        self._run_base = 0
        self.fields = {}
        self._build_ui()
        self._build_menu()
        self._connect_controls()
        self.apply_preview()
        self._refresh_controls()
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(100)  # at most 10 graphical refreshes/s
        self.render_timer.timeout.connect(self._flush_graphics)
        self.render_timer.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        heading = QLabel("PPO  /  ROVER LAB")
        heading.setObjectName("appTitle")
        root.addWidget(heading)
        subtitle = QLabel("Laser scanning  •  discrete control  •  live training preview")
        subtitle.setObjectName("muted")
        root.addWidget(subtitle)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Exactly one top-level layout for this widget.
        left = QWidget()
        left.setMinimumWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 10, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings = QWidget()
        settings_layout = QVBoxLayout(self.settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self._make_settings(settings_layout)
        scroll.setWidget(self.settings)
        left_layout.addWidget(scroll, 1)
        self.settings_note = QLabel("Set the parameters, then click Apply or Start.")
        self.settings_note.setWordWrap(True)
        self.settings_note.setObjectName("muted")
        left_layout.addWidget(self.settings_note)
        self.preview_button = QPushButton("Apply / preview environment")
        left_layout.addWidget(self.preview_button)

        speed_box = QGroupBox("Preview speed")
        speed_layout = QFormLayout(speed_box)
        self.slow_training = QCheckBox("Slow down training to inspect movement")
        self.slow_training.setToolTip("Disable for faster training; the preview may then skip frames.")
        speed_layout.addRow(self.slow_training)
        self.delay_input = QSpinBox()
        self.delay_input.setRange(10, 500)
        self.delay_input.setValue(40)
        self.delay_input.setSuffix(" ms")
        speed_layout.addRow("Step interval", self.delay_input)
        speed_note = QLabel("Policy simulation always uses this interval.\nThis controls animation timing, not physical speed.")
        speed_note.setObjectName("muted")
        speed_note.setWordWrap(True)
        speed_layout.addRow(speed_note)
        left_layout.addWidget(speed_box)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        cards = QHBoxLayout()
        self.mode_value = self._metric_card(cards, "MODE", "Ready")
        self.coverage_value = self._metric_card(cards, "TARGET COVERAGE", "0.0%")
        self.episode_value = self._metric_card(cards, "EPISODES / UPDATES", "0 / 0")
        self.cost_value = self._metric_card(cards, "LAST EPISODE COST", "—")
        right_layout.addLayout(cards)

        plots_grid = QGridLayout()
        plots_grid.setContentsMargins(0, 4, 0, 0)
        plots_grid.setColumnStretch(0, 6)
        plots_grid.setColumnStretch(1, 5)
        plots_grid.setRowStretch(0, 1)
        plots_grid.setRowStretch(1, 1)

        rover_card = QFrame()
        rover_card.setObjectName("plotCard")
        rover_layout = QVBoxLayout(rover_card)
        self.rover_figure = Figure(figsize=(6, 6), layout="constrained")
        self.rover_canvas = FigureCanvasQTAgg(self.rover_figure)
        self.rover_canvas.setMinimumSize(330, 330)
        self.rover_plot = RoverPlot(self.rover_figure)
        rover_layout.addWidget(self.rover_canvas, 1)
        self.map_caption = QLabel("Gray = target; green = scanned. The arrow shows heading.")
        self.map_caption.setWordWrap(True)
        self.map_caption.setObjectName("muted")
        rover_layout.addWidget(self.map_caption)
        self.map_toolbar = NavigationToolbar2QT(self.rover_canvas, self)
        rover_layout.addWidget(self.map_toolbar)
        plots_grid.addWidget(rover_card, 0, 0, 2, 1)

        self.metrics_tabs = QTabWidget()
        self.cost_figure = Figure(figsize=(4.5, 3), layout="constrained")
        self.cost_canvas = FigureCanvasQTAgg(self.cost_figure)
        self.cost_canvas.setMinimumSize(300, 220)
        self.cost_plot = EpisodePlot(self.cost_figure, "cost")
        self.metrics_tabs.addTab(self.cost_canvas, "Cost / episodes")
        loss_panel = QWidget()
        loss_layout = QVBoxLayout(loss_panel)
        self.loss_metric = QComboBox()
        for label, value in (("Total PPO loss", "total_loss"), ("Policy loss", "policy_loss"),
                             ("Value loss", "value_loss"), ("Policy entropy", "entropy")):
            self.loss_metric.addItem(label, value)
        loss_layout.addWidget(self.loss_metric)
        self.loss_figure = Figure(figsize=(4.5, 3), layout="constrained")
        self.loss_canvas = FigureCanvasQTAgg(self.loss_figure)
        self.loss_plot = LossPlot(self.loss_figure)
        loss_layout.addWidget(self.loss_canvas, 1)
        self.metrics_tabs.addTab(loss_panel, "PPO diagnostics")
        plots_grid.addWidget(self.metrics_tabs, 0, 1)

        self.coverage_figure = Figure(figsize=(4.5, 3), layout="constrained")
        self.coverage_canvas = FigureCanvasQTAgg(self.coverage_figure)
        self.coverage_canvas.setMinimumSize(300, 220)
        self.coverage_plot = EpisodePlot(self.coverage_figure, "coverage")
        plots_grid.addWidget(self.coverage_canvas, 1, 1)
        right_layout.addLayout(plots_grid, 1)

        hint = QLabel("Cost = negative episode return. Episode charts exclude policy simulation runs.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        right_layout.addWidget(hint)
        self.progress = QProgressBar()
        self.progress.setRange(0, 40)
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        right_layout.addWidget(self.progress)

        # Controls below all plots.
        controls = QHBoxLayout()
        self.mode_input = QComboBox()
        self.mode_input.addItem("PPO training", "train")
        self.mode_input.addItem("Policy simulation", "eval")
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.stop_button = QPushButton("Stop")
        self.reset_sim_button = QPushButton("Reset simulation")
        self.reset_train_button = QPushButton("Reset training")
        for widget in (self.mode_input, self.start_button, self.stop_button,
                       self.reset_sim_button, self.reset_train_button):
            controls.addWidget(widget)
        right_layout.addLayout(controls)
        self.deterministic = QCheckBox("Simulation: choose the most likely action (deterministic)")
        self.deterministic.setChecked(True)
        right_layout.addWidget(self.deterministic)
        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("status")
        right_layout.addWidget(self.status_label)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1065])

    def _metric_card(self, row, title, value):
        frame = QFrame()
        frame.setObjectName("metricCard")
        layout = QVBoxLayout(frame)
        name = QLabel(title)
        name.setObjectName("metricName")
        name.setWordWrap(True)
        number = QLabel(value)
        number.setObjectName("metricValue")
        layout.addWidget(name)
        layout.addWidget(number)
        row.addWidget(frame, 1)
        return number

    def _make_settings(self, parent_layout):
        defaults = Config()

        def form_for(widget):
            form = QFormLayout(widget)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            return form

        def group(title):
            box = QGroupBox(title)
            parent_layout.addWidget(box)
            return form_for(box)

        def integer(form, name, label, lo, hi, help_text=""):
            edit = QSpinBox()
            edit.setRange(lo, hi)
            edit.setValue(getattr(defaults, name))
            edit.setToolTip(help_text)
            self.fields[name] = edit
            form.addRow(label, edit)

        def number(name, lo=0, hi=500, decimals=3, step=1, factor=1,
                   suffix="", help_text=""):
            edit = QDoubleSpinBox()
            edit.setDecimals(decimals)
            edit.setRange(lo, hi)
            edit.setSingleStep(step)
            edit.setValue(getattr(defaults, name) * factor)
            edit.setSuffix(suffix)
            edit.setToolTip(help_text)
            edit.setKeyboardTracking(False)
            self.fields[name] = edit
            return edit

        def real(form, name, label, lo, hi, decimals=2, step=0.1,
                 factor=1, suffix="", help_text=""):
            form.addRow(label, number(name, lo, hi, decimals, step,
                                      factor, suffix, help_text))

        def point(form, label, x_name, y_name):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            for coordinate, name in (("x", x_name), ("y", y_name)):
                layout.addWidget(QLabel(coordinate))
                edit = number(name, help_text="Map coordinate, not a screen pixel.")
                edit.setMinimumWidth(75)
                layout.addWidget(edit, 1)
            form.addRow(label, row)

        env = group("01  Environment")
        shape = QComboBox()
        for label, value in (("Circle", "circle"), ("Ellipse", "ellipse"),
                             ("Triangle", "triangle"), ("Rectangle", "rectangle"),
                             ("Square", "square")):
            shape.addItem(label, value)
        self.fields["shape"] = shape
        env.addRow("Target shape", shape)
        integer(env, "width", "Map width", 16, 500)
        integer(env, "height", "Map height", 16, 500)
        real(env, "goal_coverage", "Coverage goal", 1, 100, 1, 1, factor=100, suffix=" %")
        integer(env, "max_steps", "Steps / episode", 10, 10000)

        geometry = group("02  Target geometry")
        note = QLabel("Coordinates use map units. Origin: bottom-left. "
                      "x points right; y points up. The target must fit inside the map.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        geometry.addRow(note)
        self.shape_stack = QStackedWidget()
        self.shape_pages = {}

        def page(key):
            widget = QWidget()
            form = form_for(widget)
            form.setContentsMargins(0, 2, 0, 2)
            self.shape_pages[key] = self.shape_stack.addWidget(widget)
            return form

        circle = page("circle")
        point(circle, "Center", "circle_cx", "circle_cy")
        real(circle, "circle_radius", "Radius", 0.001, 500, 3, 1)

        ellipse = page("ellipse")
        point(ellipse, "Center", "ellipse_cx", "ellipse_cy")
        real(ellipse, "ellipse_rx", "Radius X (a)", 0.001, 500, 3, 1,
             help_text="Horizontal semi-axis; this is a radius, not a diameter.")
        real(ellipse, "ellipse_ry", "Radius Y (b)", 0.001, 500, 3, 1,
             help_text="Vertical semi-axis; this is a radius, not a diameter.")
        ellipse.addRow(QLabel("Ellipse axes are parallel to the map axes."))

        triangle = page("triangle")
        for i in (1, 2, 3):
            point(triangle, f"Point {i}", f"triangle_x{i}", f"triangle_y{i}")
        tri_note = QLabel("Use three distinct, non-collinear points, in either order.")
        tri_note.setWordWrap(True)
        triangle.addRow(tri_note)

        rectangle = page("rectangle")
        # Rectangle and square use the same opposite-corner editor.
        self.shape_pages["square"] = self.shape_pages["rectangle"]
        point(rectangle, "Point 1", "rect_x1", "rect_y1")
        point(rectangle, "Point 2", "rect_x2", "rect_y2")
        self.corner_note = QLabel()
        self.corner_note.setWordWrap(True)
        rectangle.addRow(self.corner_note)
        geometry.addRow(self.shape_stack)

        rover = group("03  Rover and laser")
        real(rover, "beam_width", "Laser diameter", 0.1, 500, 2, 1,
             help_text="Circular footprint. Coverage includes the entire sweep between positions.")
        real(rover, "step_size", "Forward step", 0.01, 100, 2, 0.5)
        real(rover, "turn_angle_deg", "Turn in place", 0.1, 180, 1, 5, suffix=" °")
        point(rover, "Start point", "start_x", "start_y")
        self.center_start_button = QPushButton("Place start at map center")
        rover.addRow(self.center_start_button)
        real(rover, "start_theta_deg", "Start heading", -360, 360, 1, 15, suffix=" °",
             help_text="0° = +X (right); 90° = +Y (up); positive angles turn counterclockwise.")
        start_note = QLabel("The rover may start outside the target, but must stay inside the map.")
        start_note.setWordWrap(True)
        start_note.setObjectName("muted")
        rover.addRow(start_note)

        ppo = group("04  PPO learning")
        real(ppo, "lr", "Learning rate", 0.000001, 0.1, 6, 0.0001)
        real(ppo, "gamma", "Gamma", 0, 1, 3, 0.01)
        real(ppo, "gae_lambda", "GAE lambda", 0, 1, 3, 0.01)
        real(ppo, "clip_epsilon", "PPO clip", 0.001, 0.999, 3, 0.01)
        real(ppo, "entropy_coef", "Entropy coefficient", 0, 2, 4, 0.005)
        real(ppo, "value_coef", "Value coefficient", 0, 10, 3, 0.1)
        integer(ppo, "update_epochs", "Epochs / update", 1, 30)
        integer(ppo, "minibatch_size", "Minibatch", 2, 8192)
        real(ppo, "reward_scale", "PPO reward scale", 0.001, 10, 3, 0.01,
             help_text="Training only. Episode cost uses the original, unscaled rewards.")

        run = group("05  Training run")
        integer(run, "updates", "Updates per Start", 1, 10000,
                "Number of additional updates. Continue training keeps the current weights.")
        integer(run, "rollout_steps", "Steps / rollout", 8, 8192)
        integer(run, "seed", "Seed", 0, 2147483647)
        device = QComboBox()
        device.addItem("CPU", "cpu")
        device.addItem("CUDA / NVIDIA", "cuda")
        self.fields["device"] = device
        run.addRow("Device", device)
        parent_layout.addStretch(1)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        self.save_action = QAction("Save model and configuration…", self)
        self.load_action = QAction("Load model…", self)
        self.export_action = QAction("Export results to CSV…", self)
        self.export_plots_action = QAction("Save plots as PNG…", self)
        for action in (self.save_action, self.load_action, self.export_action, self.export_plots_action):
            file_menu.addAction(action)
        self.save_action.triggered.connect(self.save_checkpoint)
        self.load_action.triggered.connect(self.load_checkpoint)
        self.export_action.triggered.connect(self.export_csv)
        self.export_plots_action.triggered.connect(self.export_plots)

    def _connect_controls(self):
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.reset_train_button.clicked.connect(self.reset_training)
        self.reset_sim_button.clicked.connect(self.reset_simulation)
        self.preview_button.clicked.connect(self.apply_preview)
        self.fields["shape"].currentIndexChanged.connect(self._update_shape_page)
        self.center_start_button.clicked.connect(self._center_rover_start)
        self.mode_input.currentIndexChanged.connect(self._refresh_controls)
        self.loss_metric.currentIndexChanged.connect(self._mark_charts_dirty)
        self.slow_training.toggled.connect(self._change_delay)
        self.delay_input.valueChanged.connect(self._change_delay)
        self._update_shape_page()

    def _read_config(self):
        values = {}
        for name, field in self.fields.items():
            if isinstance(field, QComboBox):
                values[name] = field.currentData()
            elif isinstance(field, QCheckBox):
                values[name] = field.isChecked()
            else:
                values[name] = field.value()
        values["goal_coverage"] /= 100
        return Config(**values).validate()

    def _set_config_fields(self, config):
        for name, field in self.fields.items():
            value = getattr(config, name)
            if name == "goal_coverage":
                value *= 100
            if isinstance(field, QComboBox):
                field.setCurrentIndex(field.findData(value))
            elif isinstance(field, QCheckBox):
                field.setChecked(value)
            else:
                field.setValue(value)
        self._update_shape_page()

    def _update_shape_page(self, *_):
        shape = self.fields["shape"].currentData()
        self.shape_stack.setCurrentIndex(self.shape_pages[shape])
        self.corner_note.setText(
            "Opposite corners, in either order. A square requires |x2 − x1| = |y2 − y1|."
            if shape == "square" else
            "Opposite corners, in either order. Rectangle sides are parallel to the map axes.")

    @Slot()
    def _center_rover_start(self):
        self.fields["start_x"].setValue(self.fields["width"].value() / 2)
        self.fields["start_y"].setValue(self.fields["height"].value() / 2)

    def _mark_charts_dirty(self, *_):
        self._charts_dirty = True

    def _delay_ms(self, mode):
        return self.delay_input.value() if mode == "eval" or self.slow_training.isChecked() else 0

    def _change_delay(self, *_):
        if self.worker is not None:
            self.worker.set_delay(self._delay_ms(self.worker.mode))

    def _refresh_controls(self, *_):
        busy = self.worker is not None
        has_model = self.checkpoint is not None
        mode = self.mode_input.currentData()
        self.settings.setEnabled(not busy and not has_model)
        self.preview_button.setEnabled(not busy and not has_model)
        self.mode_input.setEnabled(not busy)
        self.start_button.setEnabled(not busy and (mode == "train" or has_model))
        self.start_button.setText("Continue training" if mode == "train" and has_model else "Start")
        self.stop_button.setEnabled(busy)
        self.reset_sim_button.setEnabled(not busy or self.worker.mode == "eval")
        self.deterministic.setEnabled(not busy)
        self.save_action.setEnabled(not busy and has_model)
        self.load_action.setEnabled(not busy)
        self.export_action.setEnabled(not busy and bool(self.episode_history))
        self.export_plots_action.setEnabled(not busy)
        self.settings_note.setText(
            "Parameters are locked for this session. Reset training to edit them and discard in-memory weights."
            if busy or has_model else "Set the parameters, then click Apply or Start.")

    @Slot()
    def apply_preview(self):
        if self.worker is not None:
            return
        try:
            config = self.active_config if self.checkpoint is not None else self._read_config()
            self._pending_frame = make_preview(config)
            self._flush_graphics()
        except Exception as exc:
            QMessageBox.warning(self, "Environment parameters", str(exc))

    @Slot()
    def start(self):
        if self.worker is not None:
            return
        try:
            config = self.active_config if self.checkpoint is not None else self._read_config()
            config.validate()
            if config.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable in this PyTorch installation. Select CPU.")
            # Validate geometry before launching the thread.
            preview = make_preview(config)
            mode = self.mode_input.currentData()
            if mode == "train" and preview["coverage"] >= config.goal_coverage:
                raise ValueError("The laser already meets the coverage goal at the start position. Reduce the beam or enlarge the target.")
            if mode == "eval" and self.checkpoint is None:
                raise ValueError("Train or load a model first.")
        except Exception as exc:
            QMessageBox.warning(self, "Cannot start", str(exc))
            return
        self.active_config = config
        self._run_base = len(self.update_history)
        self._pending_reset = None
        self.worker = TrainingWorker(
            config, self.checkpoint, mode=mode,
            delay_ms=self._delay_ms(mode), deterministic=self.deterministic.isChecked(), parent=self)
        self.worker.frame_ready.connect(self._receive_frame)
        self.worker.episode_ready.connect(self._receive_episode)
        self.worker.update_ready.connect(self._receive_update)
        self.worker.status_ready.connect(self.status_label.setText)
        self.worker.finished.connect(self._worker_finished)
        self.mode_value.setText("Training" if mode == "train" else "Policy simulation")
        self.progress.setRange(0, config.updates if mode == "train" else config.max_steps)
        self.progress.setValue(0)
        self.progress.setFormat("Updates: %v / %m" if mode == "train" else "Steps: %v / %m")
        self._refresh_controls()
        self.worker.start()

    @Slot()
    def stop(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopping after the current step or PPO update…")

    @Slot(object)
    def _receive_frame(self, frame):
        if self._pending_reset is None and not self._closing:
            self._pending_frame = frame

    @Slot(object)
    def _receive_episode(self, record):
        self.episode_history.append(record)
        self._charts_dirty = True

    @Slot(object)
    def _receive_update(self, record):
        self.update_history.append(record)
        self.progress.setValue(record["update"] - self._run_base)
        self._charts_dirty = True

    @Slot()
    def _flush_graphics(self):
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            self._last_frame = frame
            self.rover_plot.update(frame)
            self.rover_canvas.draw_idle()
            self.coverage_value.setText(f"{frame['coverage']:.1%}")
            self.map_caption.setText(
                f"θ = {np_degrees(frame['theta']):.1f}°  |  Episode return = {frame['reward']:.2f}\n"
                "Gray = target; green = scanned. The dashed circle is the laser footprint.")
            if self.worker is not None and self.worker.mode == "eval":
                self.progress.setValue(frame["steps"])
        if self._charts_dirty:
            self.cost_plot.update(self.episode_history)
            self.coverage_plot.update(self.episode_history)
            self.loss_plot.update(self.update_history, self.loss_metric.currentData())
            for canvas in (self.cost_canvas, self.coverage_canvas, self.loss_canvas):
                canvas.draw_idle()
            self.episode_value.setText(f"{len(self.episode_history)} / {len(self.update_history)}")
            self.cost_value.setText(f"{self.episode_history[-1]['cost']:.2f}" if self.episode_history else "—")
            self._charts_dirty = False

    @Slot()
    def _worker_finished(self):
        worker = self.worker
        if worker is None:
            return
        result, error, mode = worker.result, worker.error, worker.mode
        self.worker = None
        worker.deleteLater()
        if self._closing:
            QTimer.singleShot(0, self.close)
            return
        pending = self._pending_reset
        self._pending_reset = None
        if pending == "train":
            self._clear_training()
            return
        if result is not None and mode == "train":
            self.checkpoint = result["checkpoint"]
            self.episode_history = list(self.checkpoint["episode_history"])
            self.update_history = list(self.checkpoint["update_history"])
            stopped = result["stopped"]
            self.status_label.setText(
                "Stopped. Weights kept; partial rollout discarded. Start will begin a new episode."
                if stopped else "Training complete. Run a policy simulation or continue training.")
        elif result is not None:
            evaluation = result["evaluation"]
            actions = evaluation["actions"]
            self.status_label.setText(
                f"Test {'stopped' if result['stopped'] else 'complete'}: "
                f"coverage {evaluation['coverage']:.1%}, reward {evaluation['reward']:.2f}, "
                f"left / forward / right = {actions.get('left', 0)} / "
                f"{actions.get('forward', 0)} / {actions.get('right', 0)}. Weights unchanged.")
        if error:
            self.status_label.setText("An error occurred. See the dialog and console for details.")
            print(error, flush=True)
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Training / simulation error")
            box.setText(error.strip().splitlines()[-1])
            box.setDetailedText(error)
            box.exec()
        self.mode_value.setText("Ready")
        self._charts_dirty = True
        if pending == "sim":
            self.apply_preview()
            self.progress.setValue(0)
        self._flush_graphics()
        self._refresh_controls()

    @Slot()
    def reset_simulation(self):
        if self.worker is not None:
            if self.worker.mode == "eval":
                self._pending_reset = "sim"
                self.stop()
            return
        self.apply_preview()
        self.progress.setValue(0)
        self.status_label.setText("Simulation reset. The model and training history have been kept.")

    @Slot()
    def reset_training(self):
        if self.worker is not None or self.checkpoint is not None or self.episode_history:
            answer = QMessageBox.question(
                self, "Reset training", "Discard the in-memory model, history and training progress?\n"
                "Previously saved files will remain on disk.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        if self.worker is not None:
            self._pending_reset = "train"
            self.stop()
        else:
            self._clear_training()

    def _clear_training(self):
        self.checkpoint = None
        self.active_config = None
        self.episode_history.clear()
        self.update_history.clear()
        self._pending_frame = None
        self._charts_dirty = True
        self.mode_input.setCurrentIndex(0)
        self.mode_value.setText("Ready")
        self.progress.setValue(0)
        self.progress.setFormat("New session")
        self._refresh_controls()
        self.apply_preview()
        self._flush_graphics()
        self.status_label.setText("Training reset. Edit the parameters; new weights will be initialized on Start.")

    @Slot()
    def save_checkpoint(self):
        if self.worker is not None or self.checkpoint is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save model", "rover_ppo.pt", "PyTorch (*.pt)")
        if not path:
            return
        try:
            torch.save(self.checkpoint, path)
            self.status_label.setText(f"Saved model, optimizer, configuration and history: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    @Slot()
    def load_checkpoint(self):
        if self.worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load model", "", "PyTorch (*.pt *.pth)")
        if not path:
            return
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if "config" not in checkpoint:
                # Compatibility with PPOAgent.save() from the uploaded agent.
                config = self._read_config()
                checkpoint = dict(checkpoint, config=config.to_dict(), obs_dim=105,
                                  episode_history=[], update_history=[], format_version=1)
                QMessageBox.information(self, "Legacy model", "The file contains weights only. "
                    "The current panel settings are used. They must match the training environment.")
            else:
                config = Config.from_dict(checkpoint["config"])
            if config.device == "cuda" and not torch.cuda.is_available():
                data = config.to_dict()
                data["device"] = "cpu"
                config = Config.from_dict(data)
                checkpoint["config"] = data
            if tuple(checkpoint.get("actions", ())) != ("left", "forward", "right"):
                raise ValueError("Unsupported action order in the saved model.")
            # Validate the architecture before replacing the active session.
            from agent import ActorCritic
            model = ActorCritic(105, 3)
            model.load_state_dict(checkpoint["model_state_dict"])
            del model
            preview = make_preview(config)
            checkpoint["config"] = config.to_dict()
            checkpoint["format_version"] = 2
            self.checkpoint = checkpoint
            self.active_config = config
            self.episode_history = list(checkpoint.get("episode_history", []))
            self.update_history = list(checkpoint.get("update_history", []))
            self._set_config_fields(config)
            self._pending_frame = preview
            self._charts_dirty = True
            self._flush_graphics()
            self._refresh_controls()
            self.status_label.setText(f"Loaded: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    @Slot()
    def export_csv(self):
        if self.worker is not None or not self.episode_history:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export episodes", "episodes.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.episode_history[0]))
                writer.writeheader()
                writer.writerows(self.episode_history)
            if self.update_history:
                updates_path = Path(path).with_name(Path(path).stem + "_ppo_updates.csv")
                with updates_path.open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(self.update_history[0]))
                    writer.writeheader()
                    writer.writerows(self.update_history)
            self.status_label.setText(f"Saved CSV: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    @Slot()
    def export_plots(self):
        if self.worker is not None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select a plots folder")
        if not folder:
            return
        self._flush_graphics()
        try:
            for name, figure in (("rover", self.rover_figure), ("episode_cost", self.cost_figure),
                                 ("coverage", self.coverage_figure), ("ppo_loss", self.loss_figure)):
                figure.savefig(Path(folder) / f"{name}.png", dpi=160)
            self.status_label.setText(f"Plots saved to: {folder}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))

    def closeEvent(self, event):
        if self.worker is not None:
            # No QThread.terminate(), no blocking wait while Qt needs to process signals.
            self._closing = True
            self.worker.request_stop()
            self.status_label.setText("Finishing computation before closing…")
            self.start_button.setEnabled(False)
            self.reset_train_button.setEnabled(False)
            event.ignore()
        else:
            self.render_timer.stop()
            event.accept()


def np_degrees(theta):
    # Keep widget code lightweight; the simulator's heading is in radians.
    import math
    return math.degrees(theta)
