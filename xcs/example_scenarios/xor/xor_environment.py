# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.environment import Environment
import numpy as np
import logging


class XOREnvironment(Environment):
    def __init__(self):
        Environment.__init__(self)
        logging.info('XOR environment initialized')

        self.state = None
        self.state_length = 2
        self.possible_actions = [0, 1]
        self.step(None)

    def get_state(self):
        return self.state

    def step(self, action):
        self.state = [int(round(np.random.uniform())) for _ in range(2)]

    def reset(self):
        self.step(None)

    def print_world(self):
        print(self.state)
