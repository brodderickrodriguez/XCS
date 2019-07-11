# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.reinforcement_program import ReinforcementProgram
import logging


class MultiplexerReinforcementProgram(ReinforcementProgram):
    def __init__(self, configuration=None):
        ReinforcementProgram.__init__(self, configuration)
        logging.info('MUX ReinforcementProgram initialized')

    def determine_rho(self, sigma, action):
        self.step()
        self.end_of_program = True

        address_bits = ''.join(str(x) for x in sigma[:2])
        index_bit = int(address_bits, 2)
        data_bit_index = index_bit + len(address_bits)
        data_bit = sigma[data_bit_index]

        rho = int(data_bit == action)

        return rho

    def termination_criteria_met(self):
        return self.time_step >= self.max_steps
