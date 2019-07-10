# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

import logging
import sys
from xcs.xcs_driver import XCSDriver
from xcs.xcs import XCS
from xcs import util


def human_play_xor():
    from xcs.example_scenarios.xor.xor_environment import XOREnvironment
    from xcs.example_scenarios.xor.xor_reinforcement_program import XORReinforcementProgram

    rp = XORReinforcementProgram()
    env = XOREnvironment()
    env.human_play(reinforcement_program=rp)


def human_play_woods2():
    from xcs.example_scenarios.woods2.woods2_environment import Woods2Environment
    from xcs.example_scenarios.woods2.woods2_reinforcement_program import Woods2ReinforcementProgram

    rp = Woods2ReinforcementProgram()
    env = Woods2Environment()
    env.human_play(reinforcement_program=rp)


def xor():
    from xcs.example_scenarios.xor.xor_environment import XOREnvironment
    from xcs.example_scenarios.xor.xor_reinforcement_program import XORReinforcementProgram
    from xcs.example_scenarios.xor.xor_configuration import XORConfiguration

    driver = XCSDriver()
    driver.repetitions = 20
    driver.save_location = './xcs/example_scenarios/xor/data'
    driver.experiment_name = 'test1'

    driver.xcs_class = XCS
    driver.environment_class = XOREnvironment
    driver.reinforcement_program_class = XORReinforcementProgram
    driver.configuration_class = XORConfiguration

    driver.run()


def multiplexer():
    from xcs.example_scenarios.multiplexer.multiplexer_environment import MultiplexerEnvironment
    from xcs.example_scenarios.multiplexer.multiplexer_reinforcement_program import MultiplexerReinforcementProgram
    from xcs.example_scenarios.multiplexer.multiplexer_configuration import MultiplexerConfiguration

    driver = XCSDriver()
    driver.repetitions = 30
    driver.save_location = './xcs/example_scenarios/multiplexer/data'
    driver.experiment_name = 'test1'

    driver.xcs_class = XCS
    driver.environment_class = MultiplexerEnvironment
    driver.reinforcement_program_class = MultiplexerReinforcementProgram
    driver.configuration_class = MultiplexerConfiguration

    driver.run()


def woods2():
    from xcs.example_scenarios.woods2.woods2_environment import Woods2Environment
    from xcs.example_scenarios.woods2.woods2_reinforcement_program import Woods2ReinforcementProgram
    from xcs.example_scenarios.woods2.woods2_configuration import Woods2Configuration

    driver = XCSDriver()
    driver.repetitions = 10
    driver.save_location = './xcs/example_scenarios/woods2/data'
    driver.experiment_name = 'test1'

    driver.xcs_class = XCS
    driver.environment_class = Woods2Environment
    driver.reinforcement_program_class = Woods2ReinforcementProgram
    driver.configuration_class = Woods2Configuration

    driver.run()


if __name__ == '__main__':
    print('XCS')
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # human_play_xor()
    # human_play_woods2()

    # xor()
    # multiplexer()
    woods2()

    dir_name = './xcs/example_scenarios/woods2/data/test1'
    util.plot_results(dir_name, title='W2', interval=50)
