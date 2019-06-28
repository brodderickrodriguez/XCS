# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.reinforcement_program import ReinforcementProgram


class MultiplexerReinforcementProgram(ReinforcementProgram):
    def __init__(self):
        ReinforcementProgram.__init__(self)
        print('initialized MUX reinforcement program')

    def determine_rho(self, sigma, action):
        self.step()

        _a = (not sigma[0]) & (not sigma[1]) & sigma[2]
        _b = (not sigma[0]) & sigma[1] & sigma[3]
        _c = sigma[0] & (not sigma[1]) & sigma[4]
        _d = sigma[0] & sigma[1] & sigma[5]

        rho = int((_a or _b or _c or _d) == action)
        return rho
