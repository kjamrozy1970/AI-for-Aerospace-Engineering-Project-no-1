import random
import numpy as np
from envinronment import Env

class RLAgent:
    def __init__(self, rover):
        dimention = 100
        self.rover = rover
        self.q_tab = np.zeros((dimention**2, 4))
        self.states = np.zeros((dimention, dimention))
        self.alpha = 0.1
        self.gamma = 0.9
        self.eps = 0.2
        self.episodes = 1000
        self.actions = ["left", "up", "down", "right"]
        target = None
        self.env = Env(dimention, dimention, target)
        self.state = (0, 0)

    def choose_action(self, state):
        state_key = str(state)
        if state_key not in self.q_tab:
            self.q_tab[state_key] = np.zeros((100, 100))
        return divmod(np.argmax(self.q_tab[state_key]), 100)

    def train(self):
        for episode in range(self.episodes):
           self.env.reset()
           while True:
                if random.uniform(0, 1) < self.eps: # explore
                    action = random.choice(self.actions)
                else:   # exploit
                    action = max(self.q_tab[self.state], self.q_tab[self.state])

                if action == "left":
                    next_state = self.state[0] - 1
                if action == "right":
                    next_state = self.state[0] + 1
                if action == "up":
                    next_state = self.state[1] + 1
                if action == "down":
                    next_state = self.state[1] - 1

                _, reward, self.env.done = self.env.step(action, self.rover)
                
                error = reward + gamma * max(self.q_tab[next_state].values()) - q_tab[self.state][action]
                self.q_tab[state.state][action] += alpha * error
                
                self.state = next_state


                if self.env.done:
                    break;
                 


