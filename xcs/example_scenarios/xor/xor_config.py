# Brodderick Rodriguez
# Auburn University - CSSE
# June 18 2019

import numpy as np

max_steps_per_repetition = 10 ** 4


# the length of the classifier's condition and the length
# of the environments state
condition_length = 2

# the actions which are possible in the environment
possible_actions = [0, 1]

# create an initial population of classifiers with 
# random condition and action pairs
generate_intial_random_classifiers = False

# the maximum size of the population (in micro-classifiers)
N = 400

# learning rate for payoff, epsilon, fitness, and action_set_size
beta = 0.2

# used to calculate the fitness of a classifier
alpha = 0.1
epsilon_0 = 0.01
v = 5

# discount factor 
gamma = 0.71

# the GA threshold. GA is applied in a set when the average time
# since the last GA in the set is greater than theta_ga
theta_ga = np.random.uniform(25, 50)

# the probability of applying crossover in the GA
chi = 0.8 

# specifies the probability of mutating an allele in the offspring
mu = 0.04

# deletion threshold. If the experience of a classifier is greater 
# than theta_del, its fitness may be considered in its probability 
# of deletion
theta_del = 20

# specifies the fration of the mean fitness in population below which 
# the fitness of a classifier may be considered in its probability 
# of deletion
delta = 0.1

# subsumption threshold. experience of a classifier must be greater
# than theta_sub in order to be able to subsume another classifier
theta_sub = 20

# probability of using '#' (Classifier.WILDCARD_ATTRIBUTE_VALUE)
# in one attribute in the condition of a classifier when covering
p_sharp = 0.33

# used as initial values in new classifiers 
p_1 = 0.1
epsilon_1 = 0
F_1 = 10

# probability during action selection of choosing the 
# action uniform randomly
p_explr = 0.001

# the minimum number of actions that must be present in match_set
# or else covering will occur
# "to cause covering to provide classifiers for every action, set
# equal to number of available actions"
theta_mna = len(possible_actions)

# boolean parameter. specifies if offspring are to be tested
# for possible subsumption by parents
do_ga_subsumption = False

# boolean parameter. specifies if action sets are to be tested 
# for subsuming classifiers
do_action_set_subsumption = False
