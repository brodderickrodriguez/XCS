# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019


class ReinforcementProgram:
    DEFAULT_MAX_STEPS = 10 ** 3

    def __init__(self, configuration=None):
        print('ReinforcementProgram initialized')

        self.end_of_program = False
        self.time_step = 0

        try:
            self.max_step = configuration.max_steps
        except AttributeError:
            self.max_steps = ReinforcementProgram.DEFAULT_MAX_STEPS

    def step(self):
        self.time_step += 1
        self.end_of_program = self.time_step >= self.max_steps

    def determine_rho(self, sigma, action):
        raise NotImplementedError()

    def reset(self):
        self.end_of_program = False
        self.time_step = 0
