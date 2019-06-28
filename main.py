# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.xcs_driver import XCSDriver


from xcs.example_scenarios.xor.xor_environment import XOREnvironment
from xcs.example_scenarios.xor.xor_reinforcement_program import XORReinforcementProgram


def main():
    driver = XCSDriver()

    driver.reinforcement_program_class = XOREnvironment
    driver.environment_class = XORReinforcementProgram

    driver.run()


def human_play_xor():
    rp = XORReinforcementProgram()
    env = XOREnvironment()
    env.human_play(reinforcement_program=rp)


def plot_xor_xcs():
    from xcs import xcs_plot
    from xcs.xcs import XCS

    from xcs.example_scenarios.xor import xor_config

    env = XOREnvironment()
    rp = XORReinforcementProgram()

    xcs_object = XCS(environment=env, reinforcement_program=rp, configuration=xor_config)

    xcs_object.run_experiment(repetitions=20, print_metrics=True)

    xcs_plot.plot(xcs_object, title='XOR')


def plot_six_multiplexer_xcs():
    from xcs import xcs_plot
    from xcs.xcs import XCS

    from xcs.example_scenarios.multiplexer.multiplexer_environment import MultiplexerEnvironment
    from xcs.example_scenarios.multiplexer.multiplexer_reinforcement_program import MultiplexerReinforcementProgram
    from xcs.example_scenarios.multiplexer import multiplexer_config

    env = MultiplexerEnvironment()
    rp = MultiplexerReinforcementProgram()

    xcs_object = XCS(environment=env, reinforcement_program=rp, configuration=multiplexer_config)

    xcs_object.run_experiment(repetitions=20, print_metrics=True)

    xcs_plot.plot(xcs_object, title='6-Multiplexer')


if __name__ == '__main__':

    # plot_xor_xcs()
    plot_six_multiplexer_xcs()

    # human_play_xor()
