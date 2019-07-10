# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.reinforcement_program import ReinforcementProgram
import logging


class XORReinforcementProgram(ReinforcementProgram):
    def __init__(self, configuration=None):
        ReinforcementProgram.__init__(self, configuration)
        logging.info('XOR ReinforcementProgram initialized')

    def determine_rho(self, sigma, action):
        self.step()
        self.end_of_program = True

        rho = int(sigma[0] ^ sigma[1] == action)
        return rho

    def termination_criteria_met(self):
        return self.time_step >= self.max_steps
