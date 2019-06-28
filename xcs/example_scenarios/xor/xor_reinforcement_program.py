# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.reinforcement_program import ReinforcementProgram


class XORReinforcementProgram(ReinforcementProgram):
    def __init__(self):
        ReinforcementProgram.__init__(self)
        print('initialized XOR reinforcement program')

    def determine_rho(self, sigma, action):
        self.step()

        rho = int(sigma[0] ^ sigma[1] == action)

        return rho
