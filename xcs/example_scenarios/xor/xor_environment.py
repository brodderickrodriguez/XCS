
from xcs.environment import Environment


class XOREnvironment(Environment):
    def __init__(self):
        Environment.__init__(self)
        print('initialized XOR environment')
