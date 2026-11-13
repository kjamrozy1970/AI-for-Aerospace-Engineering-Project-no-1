"""Qt tests. Requires PySide6; can run with QT_QPA_PLATFORM=offscreen."""
import time
import pytest
pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from window import Demo


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    return application


@pytest.fixture
def window(app):
    win = Demo()
    win.fields["updates"].setValue(2)
    win.fields["rollout_steps"].setValue(64)
    win.fields["minibatch_size"].setValue(32)
    win.fields["update_epochs"].setValue(1)
    win.fields["max_steps"].setValue(16)
    win.show()
    app.processEvents()
    yield win
    if win.worker is not None:
        win.stop()
        wait_until(app, lambda: win.worker is None)
    win.close()
    app.processEvents()


def wait_until(app, predicate, seconds=15):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for Qt worker"
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def test_preview(window):
    window.apply_preview()
    assert window._last_frame["image"].shape == (100, 100)
    assert window.centralWidget().layout() is not None
    assert window.rover_canvas.figure is window.rover_figure


def test_training_gui_responsive_and_eval_keeps_weights(window, app):
    beats = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: beats.append(1))
    timer.start()
    window.slow_training.setChecked(True)
    window.delay_input.setValue(10)
    window.start()
    wait_until(app, lambda: window.worker is None)
    timer.stop()
    assert len(beats) > 10
    assert len(window.update_history) == 2
    assert len(window.episode_history) == 8
    assert window.checkpoint is not None
    original = len(window.episode_history)
    window.mode_input.setCurrentIndex(1)
    window.start()
    wait_until(app, lambda: window.worker is None)
    assert len(window.episode_history) == original
    window.reset_simulation()
    assert window.checkpoint is not None
    assert window._last_frame["steps"] == 0


def test_stop_and_safe_close(window, app):
    window.slow_training.setChecked(True)
    window.delay_input.setValue(100)
    window.start()
    QTimer.singleShot(50, window.stop)
    wait_until(app, lambda: window.worker is None)
    assert window.checkpoint is not None
    window.start()
    window.close()
    wait_until(app, lambda: window.worker is None)


@pytest.mark.parametrize("shape", ["circle", "ellipse", "triangle", "rectangle", "square"])
def test_shape_page_config_and_preview(window, app, shape):
    combo = window.fields["shape"]
    combo.setCurrentIndex(combo.findData(shape))
    window.fields["start_x"].setValue(10)
    window.fields["start_y"].setValue(15)
    window.apply_preview()
    app.processEvents()
    assert window.shape_stack.currentIndex() == window.shape_pages[shape]
    assert window._last_frame["shape"] == shape
    assert window._last_frame["x"] == 10
    assert window._last_frame["y"] == 15
    assert len(window.rover_plot.outline.get_xdata()) >= 4


def test_custom_fields_reach_environment_without_recentring(window):
    combo = window.fields["shape"]
    combo.setCurrentIndex(combo.findData("ellipse"))
    for name, value in dict(ellipse_cx=35, ellipse_cy=60, ellipse_rx=15, ellipse_ry=8,
                            start_x=12.25, start_y=14.75).items():
        window.fields[name].setValue(value)
    window.apply_preview()
    frame = window._last_frame
    assert frame["x"] == 12.25 and frame["y"] == 14.75
    assert frame["target_outline"][:, 0].min() == pytest.approx(20)
    assert frame["target_outline"][:, 1].max() == pytest.approx(68)


def test_center_start_button_writes_explicit_coordinates(window):
    window.fields["width"].setValue(80)
    window.fields["height"].setValue(60)
    window.center_start_button.click()
    assert window.fields["start_x"].value() == 40
    assert window.fields["start_y"].value() == 30


def test_english_controls_and_configuration_restore(window):
    from config import Config
    cfg = Config(shape="triangle", triangle_x1=15, triangle_y1=15,
                 triangle_x2=85, triangle_y2=20, triangle_x3=35, triangle_y3=75,
                 start_x=5, start_y=7)
    window._set_config_fields(cfg)
    assert window._read_config() == cfg
    assert window.start_button.text() == "Start"
    assert window.reset_sim_button.text() == "Reset simulation"
    assert window.reset_train_button.text() == "Reset training"
    assert window.metrics_tabs.tabText(0) == "Cost / episodes"


def test_invalid_square_shows_english_error_and_keeps_preview(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text:
                        messages.append((title, text)))
    before = window._last_frame
    shape = window.fields["shape"]
    shape.setCurrentIndex(shape.findData("square"))
    window.fields["rect_x2"].setValue(70)
    window.fields["rect_y2"].setValue(80)
    window.apply_preview()
    assert messages and messages[0][0] == "Environment parameters"
    assert "For a square" in messages[0][1]
    assert window._last_frame is before


def test_geometry_and_position_preserved_when_loading_checkpoint(window, monkeypatch, tmp_path):
    from config import Config
    from engine import train_session
    from PySide6.QtWidgets import QFileDialog
    import torch
    config = Config(shape="ellipse", ellipse_cx=40, ellipse_cy=60,
                    ellipse_rx=20, ellipse_ry=10, start_x=10, start_y=15,
                    updates=1, rollout_steps=32, minibatch_size=16, max_steps=16,
                    update_epochs=1)
    checkpoint = train_session(config)["checkpoint"]
    path = tmp_path / "gui_model.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))
    window.load_checkpoint()
    assert window.active_config == config
    assert window.fields["ellipse_cx"].value() == 40
    assert window.fields["start_x"].value() == 10
    assert window._last_frame["shape"] == "ellipse"
    assert not window.settings.isEnabled()


def test_no_layout_warnings_when_constructing_window(app):
    from PySide6.QtCore import qInstallMessageHandler
    messages = []
    previous = qInstallMessageHandler(lambda kind, ctx, msg: messages.append(msg))
    win = None
    try:
        win = Demo()
        app.processEvents()
        assert not any("QLayout" in msg for msg in messages)
    finally:
        if win is not None:
            win.close()
        qInstallMessageHandler(previous)
