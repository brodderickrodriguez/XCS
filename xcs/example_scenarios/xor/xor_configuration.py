# Brodderick Rodriguez
# Auburn University - CSSE
# June 29 2019

from xcs.configuration import Configuration


class XORConfiguration(Configuration):
    def __init__(self):
        Configuration.__init__(self)

        self.max_steps_per_repetition = 10000

        # the maximum size of the population (in micro-classifiers)
        self.N = 400

        # learning rate for payoff, epsilon, fitness, and action_set_size
        self.beta = 0.2

        # used to calculate the fitness of a classifier
        self.alpha = 0.1
        self.epsilon_0 = 0.01
        self.v = 5

        # discount factor
        self.gamma = 0.9

        # the GA threshold. GA is applied in a set when the average time
        # since the last GA in the set is greater than theta_ga
        self.theta_ga = 25

        # used as initial values in new classifiers
        self.p_1 = 0.1
        self.epsilon_1 = 0
        self.F_1 = 0.1

        # probability during action selection of choosing the
        # action uniform randomly
        self.p_explr = 0.01

        self.theta_mna = 2

        # boolean parameter. specifies if offspring are to be tested
        # for possible subsumption by parents
        self.do_ga_subsumption = True

        # boolean parameter. specifies if action sets are to be tested
        # for subsuming classifiers
        self.do_action_set_subsumption = False
