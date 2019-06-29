# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.xcs_driver import XCSDriver

import logging
import sys


def human_play_xor():
    from xcs.example_scenarios.xor.xor_environment import XOREnvironment
    from xcs.example_scenarios.xor.xor_reinforcement_program import XORReinforcementProgram

    rp = XORReinforcementProgram()
    env = XOREnvironment()
    env.human_play(reinforcement_program=rp)


def plot_xor_xcs():
    from xcs import xcs_plot
    from xcs.xcs import XCS

    from xcs.example_scenarios.xor.xor_environment import XOREnvironment
    from xcs.example_scenarios.xor.xor_reinforcement_program import XORReinforcementProgram
    from xcs.example_scenarios.xor import xor_config

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data = {'rhos': [], 'predicted_rhos': [], 'microclassifier_counts': []}

    for _ in range(10):
        logging.info('starting repetition...')
        env = XOREnvironment()
        rp = XORReinforcementProgram(configuration=xor_config)

        xcs_object = XCS(environment=env, reinforcement_program=rp, configuration=xor_config)
        xcs_object.run_experiment()

        for key, val in xcs_object.metrics_history.items():
            data[key].append(val)

    xcs_plot.plot2(data, title='XOR')


def plot_six_multiplexer_xcs():
    from xcs import xcs_plot
    from xcs.xcs import XCS

    from xcs.example_scenarios.multiplexer.multiplexer_environment import MultiplexerEnvironment
    from xcs.example_scenarios.multiplexer.multiplexer_reinforcement_program import MultiplexerReinforcementProgram
    from xcs.example_scenarios.multiplexer import multiplexer_config

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data = {'rhos': [], 'predicted_rhos': [], 'microclassifier_counts': []}

    for _ in range(10):
        logging.info('starting repetition...')
        env = MultiplexerEnvironment()
        rp = MultiplexerReinforcementProgram(configuration=multiplexer_config)

        xcs_object = XCS(environment=env, reinforcement_program=rp, configuration=multiplexer_config)
        xcs_object.run_experiment()

        for key, val in xcs_object.metrics_history.items():
            data[key].append(val)

    xcs_plot.plot2(data, title='6-Multiplexer')


if __name__ == '__main__':

    plot_xor_xcs()
    # plot_six_multiplexer_xcs()

    # human_play_xor()
