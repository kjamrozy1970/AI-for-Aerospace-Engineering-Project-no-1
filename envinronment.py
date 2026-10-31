import numpy as np

class Env:
    def __init__(self, width, height, target):
        self.width = width
        self.height = height
        self.board = np.zeros((self.width, self.height))
        self.done = False
        self.max_steps = 1000
        self.steps = 0

    def reset(self):
        self.board = np.zeros((self.width, self.height))
        self.steps = 0 
        self.done = False
        return self.board

    def step(self, action, rover):
        x, y = action
        self.steps += 1
        if self.board[x, y] == 0:
            self.board[x, y] = rover
            reward, self.done = self.check(rover)
            return self.board, reward, self.done
        return self.board, -1, self.done

    def check(self, rover_x, rover_y, target):
        for i in range(self.max_steps):
            if target.patch.contains_point((rover_x, rover_y), 1):
                return 10, True
            if self.steps == self.max_steps:
                return 0, True
            return 0, False

