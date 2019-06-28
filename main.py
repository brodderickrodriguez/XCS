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


if __name__ == '__main__':
    main()
