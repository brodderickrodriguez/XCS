# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.xcs import XCS


class XCSDriver:
    def __init__(self):
        print('driver initialized')

        self.xcs_object = None
        self.reinforcement_program_class = None
        self.environment_class = None

    def run(self):
        env = self.environment_class()
        rp = self.reinforcement_program_class()
        self.xcs_object = XCS(_env=env, _rp=rp)
