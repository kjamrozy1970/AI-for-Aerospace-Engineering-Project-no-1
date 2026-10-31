from rover import Rover
from agent import RLAgent
from envinronment import Env

import os

from matplotlib import pyplot as plt
import numpy as np

# test
rover = Rover(0, 0, 0, False)
rover.move(1, 1)
rover.move(1, 0)
rover.move(0, 1)

plane = plt.Circle((40, 60), 30, color = "blue", alpha = 0.2, label = "circle workspace")
trian = plt.Polygon(((1,0), (0,1), (1,1)), color = "green", alpha = 0.2, label = "triangle workspace")

agent = RLAgent(rover)
agent.train()

plt.figure("Rover path")
plt.gca().add_patch(plane)
plt.gca().add_patch(trian)
plt.plot(rover.x_list, rover.y_list, label = "rover movement")

plt.xlim(0,100)
plt.ylim(0,100)
plt.title("Rover movment")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

