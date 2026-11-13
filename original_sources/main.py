from envinronment import Env
from agent import PPOAgent
from window import Demo
from PySide6.QtWidgets import QApplication
import sys
import random
import numpy as np
import torch


def main():

    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    app = QApplication(sys.argv)
    window = Demo()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


