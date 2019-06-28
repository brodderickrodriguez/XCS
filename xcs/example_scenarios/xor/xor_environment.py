# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.environment import Environment
import numpy as np


class XOREnvironment(Environment):
    def __init__(self):
        Environment.__init__(self)
        print('initialized XOR environment')

        self.state = None
        self.state_length = 0
        self.step(None)

    def get_state(self):
        return self.state

    def step(self, action):
        self.state = [int(round(np.random.uniform())) for _ in range(2)]

    def reset(self, action):
        self.step(None)

    def print_world(self):
        print(self.state)
