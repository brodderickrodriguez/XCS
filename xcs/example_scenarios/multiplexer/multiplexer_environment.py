# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.environment import Environment
import numpy as np


class MultiplexerEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self)
        print('initialized MUX environment')

        self.state = None
        self.state_length = 6
        self.step(None)

    def get_state(self):
        return self.state

    def step(self, action):
        self.state = [int(round(np.random.uniform())) for _ in range(self.state_length)]

    def reset(self):
        self.step(None)

    def print_world(self):
        print(self.state)
