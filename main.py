"""Run: python main.py"""
import sys
import torch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QLocale
from window import Demo


def main():
    # Small MLP: avoid using all CPU cores and starving the Qt event loop.
    # Set before constructing a model or running autograd.
    torch.set_num_threads(1)
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    app = QApplication(sys.argv)
    app.setApplicationName("PPO Rover Lab")
    app.setStyle("Fusion")
    window = Demo()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
