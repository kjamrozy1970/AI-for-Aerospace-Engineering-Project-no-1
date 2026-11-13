"""Qt bridge. No widget, figure or Matplotlib artist is touched in run()."""
import traceback
from PySide6.QtCore import QThread, Signal
from engine import RunControl, train_session, evaluate_session


class TrainingWorker(QThread):
    frame_ready = Signal(object)
    episode_ready = Signal(object)
    update_ready = Signal(object)
    status_ready = Signal(str)

    def __init__(self, config, checkpoint=None, *, mode="train", delay_ms=0,
                 deterministic=True, parent=None):
        super().__init__(parent)
        self.config = config
        self.checkpoint = checkpoint
        self.mode = mode
        self.deterministic = deterministic
        self.control = RunControl(delay_ms)
        self.result = None
        self.error = None

    def request_stop(self):
        # This method can be called from the GUI. It only sets a Python Event;
        # a queued stop slot would not run during a long training loop.
        self.control.stop()

    def set_delay(self, milliseconds):
        self.control.set_delay(milliseconds)

    def run(self):
        try:
            if self.mode == "train":
                self.result = train_session(
                    self.config, self.checkpoint, control=self.control,
                    on_frame=self.frame_ready.emit,
                    on_episode=self.episode_ready.emit,
                    on_update=self.update_ready.emit,
                    on_status=self.status_ready.emit,
                )
            else:
                self.result = evaluate_session(
                    self.config, self.checkpoint, control=self.control,
                    deterministic=self.deterministic,
                    on_frame=self.frame_ready.emit,
                    on_status=self.status_ready.emit,
                )
        except Exception:
            self.error = traceback.format_exc()
