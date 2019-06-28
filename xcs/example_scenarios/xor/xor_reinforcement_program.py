
from xcs.reinforcement_program import ReinforcementProgram


class XORReinforcementProgram(ReinforcementProgram):
    def __init__(self):
        ReinforcementProgram.__init__(self)
        print('initialized XOR reinforcement program')
