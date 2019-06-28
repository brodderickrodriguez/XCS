# Brodderick Rodriguez
# Auburn University - CSSE
# June 6 2019

import time
import logging
import operator
import numpy as np
from xcs.classifier import Classifier
import xcs.xcs_metrics as metrics
import xcs.default_config


class XCS:
    def __init__(self, environment, reinforcement_program, configuration=None):
        # if there is no configuration specified
        if configuration is None:
            # get the basic default configuration
            config_dict = self._basic_configuration_dict()
        else:
            # otherwise get the specified configuration
            config_dict = configuration.__dict__

        # for each key and default value in the basic config
        for key, default_val in self._basic_configuration_dict().items():
            # try to get this key value form the config dictionary
            try:
                self.__dict__[key] = config_dict[key]

            # if the key does not exist (i.e. was not specified in
            # the configuration), then grab its default value
            except KeyError:
                self.__dict__[key] = default_val

        # all the classifiers that currently exist
        self.population = []

        # formed from population. all classifiers that their
        # condition matches the current state
        self.match_set = []

        # formed from match_set. all classifiers that propose
        # the action which was committed
        self.action_set = []

        # the action_set which was active at the previous time_step
        self.previous_action_set = []

        # the environment object
        self.env = environment

        # the reinforcement program object
        self.rp = reinforcement_program

        # dictionary containing all the seen rewards, expected rewards,
        # states, actions, and the number of microclassifiers
        self.metrics_history = {'rhos': [],
                                'predicted_rhos': [],
                                'sigmas': [],
                                'actions': [],
                                'microclassifier_counts': []}

        # if the configuration speficies creating an initial population
        # of classifiers then call the subroutine that performs it
        if self.generate_intial_random_classifiers:
            self._generate_initial_random_classifiers()

    def _basic_configuration_dict(self):
        basic_config_dict = {
            # the length of the classifier's condition and the length
            # of the environments state
            'condition_length': 2,

            # the actions which are possible in the environment
            'possible_actions': [0, 1],

            # create an initial population of classifiers with
            # random condition and action pairs
            'generate_intial_random_classifiers': False,

            # the maximum size of the population (in micro-classifiers)
            'N': 400,

            # learning rate for payoff, epsilion, fitness, and action_set_size
            'beta': np.random.uniform(0.1, 0.9),

            # used to calculate the fitness of a classifier
            'alpha': 0.1,
            'epsilon_0': 0.01,
            'v': 5,

            # discount factor
            'gamma': np.random.uniform(0.9, 0.99),

            # the GA threshold. GA is applied in a set when the average time
            # since the last GA in the set is greater than theta_ga
            'theta_ga': np.random.uniform(25, 50),

            # the probability of applying crossover in the GA
            'chi': np.random.uniform(0.5, 1.0),

            # specifies the probability of mutating an allele in the offspring
            'mu': np.random.uniform(0.01, 0.05),

            # deletion threshold. If the experience of a classifier is greater
            # than theta_del, its fitness may be considered in its probability
            # of deletion
            'theta_del': 20,

            # specifies the fration of the mean fitness in population below which
            # the fitness of a classifier may be considered in its probability
            # of deletion
            'delta': 0.1,

            # subsumption threshold. experience of a classifier must be greater
            # than theta_sub in order to be able to subsume another classifier
            'theta_sub': 20,

            # probability of using '#' (Classifier.WILDCARD_ATTRIBUTE_VALUE)
            # in one attribute in the condition of a classifier when covering
            'p_sharp': 0.33,

            # used as initial values in new classifiers
            'p_1': np.random.uniform(0, 10 ** -4),
            'epsilon_1': np.random.uniform(0, 10 ** -4),
            'F_1': np.random.uniform(0, 10 ** -4),

            # probability during action selection of choosing the
            # action uniform randomly
            'p_explr': 0.0001,

            # the minimum number of actions that must be present in match_set
            # or else covering will occur
            # "to cause covering to provide classifiers for every action, set
            # equal to number of available actions"
            'theta_mna': len([0, 1]),

            # boolean parameter. specifies if offspring are to be tested
            # for possible subsumption by parents
            'do_ga_subsumption': True,

            # boolean parameter. specifies if action sets are to be tested
            # for subsuming classifiers
            'do_action_set_subsumption': False}

        return basic_config_dict

    def _generate_initial_random_classifiers(self):
        """ generate an initial population of classifiers by
            setting each condition and action to a random boolean var

        Parameters:
            None

        Returns:
            None
        """
        for _ in range(self.N):
            # create a new classifier
            cl = Classifier(xcs_object=self)

            # set the classifier's condition to random chocie
            cl.condition = list(np.random.choice(
                Classifier.POSSIBLE_ATTRIBUTE_VALUES,
                self.condition_length))

            # set the classifier's action to random [0, 1]
            cl.action = round(np.random.random())

            # add the classifier to the population
            self.population.append(cl)

    def run_experiment(self, repetitions=1, print_metrics=False,
                       reset_classifiers=True):
        for repetition in range(repetitions):
            self.env.reset()
            self.rp.reset()

            if reset_classifiers:
                self.population = []
                self.match_set = []
                self.action_set = []
                self.previous_action_set = []

            repetition_metrics = self._run_experiment(print_metrics)

            for key, value in repetition_metrics.items():
                self.metrics_history[key].append(value)

    def _run_experiment(self, print_metrics=False):
        np.random.seed(int(time.time()))
        logging.info('starting experiment...')
        rho_neg_1 = 0
        sigma_neg_1 = []

        repetition_metrics = {'rhos': [],
                              'predicted_rhos': [],
                              'sigmas': [],
                              'actions': [],
                              'microclassifier_counts': []}

        while not self.rp.end_of_program:
            # print metrics every 5000 steps
            if print_metrics and self.rp.time_step % 5 == 1:
                metrics.print_best_classifiers(self, n=5)
            # metrics.print_mean_reward(self, time_frame=100)

            # get current situation from environment
            sigma = self.env.get_state()
            logging.debug('sigma = {}'.format(sigma))

            # generate match set. uses population and sigma
            self.match_set = self.generate_match_set(sigma)
            logging.debug('match_set = {}'.format(self.match_set))

            # generate prediction dictionary
            predictions = self.generate_prediction_dictionary()
            logging.debug('predictions = {}'.format(predictions))

            # select action using predictions
            action = self.select_action(predictions)
            logging.debug('selected action = {}'.format(action))

            # generate action set using action and match_set
            self.action_set = self.generate_action_set(action)
            logging.debug('action_set = {}'.format(self.action_set))

            # commit action
            self.env.step(action)

            # get payoff for committing this action
            rho = self.rp.determine_rho(sigma, action)
            logging.debug('payoff (rho) = {}'.format(rho))

            # save the predicted payoff from committing this action
            repetition_metrics['predicted_rhos'].append(predictions[action])

            # save the actual payoff recieved for committing this action
            repetition_metrics['rhos'].append(rho)

            # save the state at this time step
            repetition_metrics['sigmas'].append(sigma)

            # save the action taken at this time step
            repetition_metrics['actions'].append(action)

            # save the number of microclassifiers at this time step
            num_micro_classifiers = sum([cl.numerosity for cl in self.population])

            repetition_metrics['microclassifier_counts']. \
                append(num_micro_classifiers)

            # if previous_action_set is not empty
            if len(self.previous_action_set) > 0:
                # compute deletion probability
                # P = rho_neg_1 * self.gamma * max(predictions.keys())
                P = (1 - self.alpha) * rho_neg_1 + self.alpha * self.gamma * max(predictions.keys())

                # update previous_action_set by using
                # P probability of deletion
                self.update_set(A=self.previous_action_set, P=P)

                # run genetic algorithm on previous_action_set and
                # previous_sigma inserting and possibly deleting in population
                self.run_GA(self.previous_action_set, sigma_neg_1)

            # if experiment is over based on information from
            # reinforcement program
            if self.rp.end_of_program:
                P = rho

                # update action_set by using
                # P probability of deletion
                self.update_set(A=self.action_set, P=rho)

                # run genetic algorithm on previous_action_set and
                # previous_sigma inserting and possibly deleting in population
                self.run_GA(self.action_set, sigma)

                # empty previous_action_set
                self.previous_action_set = []
            else:
                # update previous_action_set
                self.previous_action_set = self.action_set

                # update previous rho
                rho_neg_1 = rho

                # update previous sigma
                sigma_neg_1 = sigma

        return repetition_metrics

    def generate_match_set(self, sigma):
        # local variable to hold all matching classifiers
        M = []

        # continue until we have at least one classifer that matches sigma
        while len(M) == 0:
            # iterate over all classifiers
            for cl in self.population:
                # check if each classifier matches the current situation (sigma)
                if self.does_match(cl, sigma):
                    # if the classifier matches, add it to the new match set
                    M.append(cl)

            # collect all the unique actions found in the local match set
            all_found_actions = set([cl.action for cl in M])

            # if the length of all unique actions is less than our
            # threshold, theta_mna, begin covering procedure
            if len(all_found_actions) < self.theta_mna:
                # create a new classifier, cl_c
                # using the local match set and the current situation (sigma)
                cl_c = self.generate_covering_classifier(M, sigma)

                # add the new classifier cl_c to the population
                self.population.append(cl_c)

                # choose individual classifiers by roulette-wheel
                # selection for deletion
                self.delete_from_population()

                # empty local match set M
                M = []

        return M

    def does_match(self, cl, sigma):
        # for each attribute in cl and sigma
        for i in range(len(sigma)):
            # get local references to attributes to be compared
            xi = cl.condition[i]
            yi = sigma[i]

            # if x is not wildcard match and x != y
            if xi != Classifier.WILDCARD_ATTRIBUTE_VALUE and xi != yi:
                return False

        # if the for checks all attributes and each passes fail test
        return True

    def generate_covering_classifier(self, M, sigma):
        # initialize new classifier
        cl = Classifier(xcs_object=self)

        # for each attribute in cl's condition
        for i in range(self.condition_length):
            # if a random number is less than the probsability of assiging
            # a wildcard '#'
            if np.random.uniform() < self.p_sharp:
                # assign it to a wildcard '#'
                cl.condition[i] = Classifier.WILDCARD_ATTRIBUTE_VALUE
            else:
                # otherwise, match the condition attribute in sigma
                cl.condition[i] = sigma[i]

        # assign a random action to this classifier that is not
        # found in the match_set
        # get all the unique actions found in the match_set
        actions_found = set([cl.action for cl in self.match_set])

        # subtract the possible actions from the actions found
        difference_actions = set(self.possible_actions) - actions_found

        # if there are possible actions that are not in the actions_found
        if len(difference_actions) > 0:
            # find a random index in difference_actions
            rand_idx = int(np.floor(np.random.uniform() *
                                    len(difference_actions)))

            # set the action to the action corresponding to the random index
            cl.action = list(difference_actions)[rand_idx]
        else:
            # find a random index in the possible actions
            rand_idx = int(np.floor(np.random.uniform() *
                                    len(self.possible_actions)))

            # set the action to the action corresponding to the random index
            cl.action = self.possible_actions[rand_idx]

        # set the time step to the current time step
        cl.time_step = self.rp.time_step

        # set the numerosity to 1 because this method only gets called when
        # there are insufficient classifier actions
        cl.numerosity = 1

        # set the action_set_size to 1 because this method only gets called when
        # there are insufficient classifier actions
        cl.action_set_size = 1

        return cl

    def generate_prediction_dictionary(self):
        # initialize the prediction dictionary
        PA = {a: 0.0 for a in self.possible_actions}

        # initialize the fitness sum dictionary
        FSA = {a: 0.0 for a in self.possible_actions}

        # for each classifier in match_set
        for i in range(len(self.match_set)):
            # set a local variable
            cl = self.match_set[i]

            # if the value in prediction dictonary for cl.action is None
            if PA[cl.action] is None:
                # set it by accounting for fitness and predicted_payoff
                PA[cl.action] = cl.predicted_payoff * cl.fitness
            else:
                # otherwise add to the action's weighted average
                PA[cl.action] += cl.predicted_payoff * cl.fitness

            # add to the action's fitness sum
            FSA[cl.action] += cl.fitness

        # for each poissible action
        for action in PA.keys():
            # if the fitness sum of the action is not zero
            if FSA[action] != 0:
                # divide by the sum of the fitness for action across
                # all classifiers
                PA[action] /= FSA[action]

        return PA

    def select_action(self, predictions):
        # select action accoring to an epsilon-greedy policy
        if np.random.uniform() < self.p_explr:
            logging.debug('selecting random action...')

            # do pure exploration
            # find a random index in the list of self.possible_actions
            rand_idx = int(np.floor(np.random.uniform() *
                                    len(self.possible_actions)))

            # then return the action that corresponds with that index
            return self.possible_actions[rand_idx]
        else:
            # otherwise, return the best action to take
            best_action = max(predictions.items(), key=operator.itemgetter(1))

            # return the action corresponding to the highest weighted payoff
            return best_action[0]

    def generate_action_set(self, action):
        # initialize _action_set to an empty list
        _action_set = []

        # for each classifier in the match_set
        for cl in self.match_set:
            # if the classifier suggests the chosen action
            if cl.action == action:
                # add it to the action_set
                _action_set.append(cl)

        return _action_set

    def update_set(self, A, P):
        # * equations found on page 12 of 'An Algorithmic Description of XCS' *
        # for each classifier in A
        # (either self.action_set or self.previous_action_set)
        for cl in A:
            # update experience
            cl.experience += 1

            # if classifier experience is less than inverse of learning rate
            # used to determine update methods for predicted_payoff (p),
            # error (epsilon), and action_set_size (as)
            cl_exp_under_threashold = cl.experience < (1 / self.beta)

            # the difference between the p-ty any predicted payoff
            # used to update payoff and error
            payoff_difference = P - cl.predicted_payoff

            # the sum of the differences between each classifier's
            # numerosity and the action set size
            # used to update action set size
            summed_difference = np.sum([c.numerosity - cl.action_set_size for c in A])

            # update predicted_payoff (p)
            if cl_exp_under_threashold:
                cl.predicted_payoff += payoff_difference / cl.experience
            else:
                cl.predicted_payoff += self.beta * payoff_difference

            # update prediction error (epsilon)
            if cl_exp_under_threashold:
                cl.epsilon += (np.abs(payoff_difference) - cl.epsilon) / cl.experience
            else:
                cl.epsilon += self.beta * (np.abs(payoff_difference) - cl.epsilon)

            # update action_set_size (as)
            if cl_exp_under_threashold:
                cl.action_set_size += summed_difference / cl.experience
            else:
                try:
                    cl.action_set_size += self.beta * summed_difference
                except:
                    print(self.beta, summed_difference, type(summed_difference))
                    f = open('./.debug1', 'w')
                    s = str(cl.action_set_size) + '\n' + \
                        str(self.beta) + '\n' + str(self.summed_difference) + \
                        '\n' + str(type(summed_difference))
                    f.write(s)
                    f.close()

        # update fitness for each classifier in A
        self.update_fitness(A)

        # if the program is using action_set_subsumption then
        # call the procedure
        if self.do_action_set_subsumption:
            self.do_action_set_subsumption_procedure(A)

    def update_fitness(self, set_):
        # set a local variable to track the accuracy over the entire set_
        accuracy_sum = 0.0

        # initialize accuracy vector (in dictionary form)
        k = {set_[i]: 0.0 for i in range(len(set_))}

        # for each classifier in set_
        for cl in set_:
            # if classifier error is less than the error threashold
            if cl.epsilon < self.epsilon_0:
                # set the accuracy to 1 (100%)
                k[cl] = 1
            else:
                k[cl] = np.power((cl.epsilon / self.epsilon_0), -self.v) \
                        * self.alpha

            # update accuracy_sum using a weighted sum based on
            # classifier numerosity
            accuracy_sum += k[cl] * cl.numerosity

        # for each classifier in set_
        for cl in set_:
            cl.fitness += self.beta * \
                          (((k[cl] * cl.numerosity) / accuracy_sum) - cl.fitness)

    def run_GA(self, set_, sigma):
        # get average time since last GA
        weighted_time = sum([cl.last_time_step * cl.numerosity for cl in set_])

        # get the total number of micro-classifiers currently present in set_
        num_micro_classifiers = sum([cl.numerosity for cl in set_])

        # compute the average time since last GA
        average_time = weighted_time / num_micro_classifiers

        # if the average time since last GA is less than the threashold
        # then do nothing
        if self.rp.time_step - average_time <= self.theta_ga:
            return

        # update the time since last GA for all classifiers
        for cl in set_:
            cl.last_time_step = self.rp.time_step

        # select two parents from the set_
        parent1 = self.select_offspring(set_)
        parent2 = self.select_offspring(set_)

        # copy each parent and create two new classifiers, child1 and child2
        child1 = parent1.copy()
        child2 = parent2.copy()

        # set their numerosity to 1
        child1.numerosity = 1
        child2.numerosity = 1

        # set their experience to 0
        child1.experience = 0
        child2.experience = 0

        # if a random number is less than the threashold for applying crossover
        if np.random.uniform() < self.chi:
            # apply crossover to child1 and child2
            self.apply_crossover(child1, child2)

            # set child1's payoff to the mean of both parents
            child1.predicted_payoff = np.mean([parent1.predicted_payoff,
                                               parent2.predicted_payoff])

            # set child1's error (epsilon) to the mean of both parents
            child1.epsilon = np.mean([parent1.epsilon, parent2.epsilon])

            # set child1's fitness to the mean of both parents
            child1.fitness = np.mean([parent1.fitness, parent2.fitness])

            # set child2's payoff to child1's payoff
            child2.predicted_payoff = child1.predicted_payoff

            # set child2's epsilon (error) to child1's epsilon (error)
            child2.epsilon = child1.epsilon

            # set child2's fitness to child1's fitness
            child2.fitness = child1.fitness

        # set child1's fitness to 10% of it parents value to verify the
        # classifier's worthiness
        child1.fitness *= 0.1

        # set child2's fitness to 10% of it parents value to verify the
        # classifier's worthiness
        child2.fitness *= 0.1

        # for both children
        for child in [child1, child2]:
            # apply mutation to child according to sigma
            self.apply_mutation(child, sigma)

            # if subsumption is true
            if self.do_ga_subsumption:
                # check if parent1 subsumes child
                if self.does_subsume(parent1, child):
                    # if it does, increment parent1 numerosity
                    parent1.numerosity += 1
                # check if parent2 subsumes child
                elif self.does_subsume(parent2, child):
                    # if it does, increment parent2 numerosity
                    parent2.numerosity += 1
                else:
                    # otherwise, add the child to the population of classifiers
                    self.insert_in_population(child)
            else:
                # if subsumption is false, add the child
                # to the population of classifiers
                self.insert_in_population(child)

            # choose individual classifiers by roulette-wheel
            # selection for deletion
            self.delete_from_population()

    def select_offspring(self, set_):
        # set a local variable to track the fitness over the entire set_
        fitness_sum = 0

        # for each classifier in the set_
        for cl in set_:
            # add its fitness to the fitness_sum
            fitness_sum += cl.fitness

        # select a random threashold for fitness_sum
        choice_point = np.random.uniform() * fitness_sum

        # reset fitness_sum to zero
        fitness_sum = 0

        # for each classifier in the set_
        for cl in set_:
            # add its fitness to the fitness_sum
            fitness_sum += cl.fitness

            # if we pass the choice_point, return the classifier
            # which cause us to pass the threashold
            if fitness_sum > choice_point:
                return cl

    def apply_crossover(self, child1, child2):
        # set a local variable for some random index in which we
        # terminate the while loop
        x = np.random.uniform() * (len(child1.condition) + 0)

        # set a local variable for some random index in which we
        # terminate the while loop
        y = np.random.uniform() * (len(child2.condition) + 0)

        # set a local pointer for the while loop
        i = 0

        # if x is greater than y
        if x > y:
            # then swap their values
            x, y = y, x

        # pythonic do-while i > y
        while True:
            # while we are within the random bounds specified by x and y
            if x <= i and i < y:
                # swap the i-th condition in child1's and child2's condition
                child1.condition[i], child2.condition[i] = \
                    child2.condition[i], child1.condition[i]

            # increment the do-while loop pointer
            i += 1

            # if our do-while criteria fails, break out of the loop
            if i > y:
                break

    def apply_mutation(self, child, sigma):
        # for each index in the child's condition
        for i in range(self.condition_length):
            # if some random number is less than
            # the probability of mutating an allele in the offspring
            if np.random.uniform() < self.mu:
                # if the attribute at index i is already the wildcard
                if child.condition[i] == Classifier.WILDCARD_ATTRIBUTE_VALUE:
                    # swap it with the i-th attribute in sigma
                    child.condition[i] = sigma[i]
                else:
                    # otherwise, swap it to the wildcard
                    child.condition[i] = Classifier.WILDCARD_ATTRIBUTE_VALUE

        # if some random number is less than
        # the probability of mutating an allele in the offspring
        if np.random.uniform() < self.mu:
            # then generate a list of all the other possible actions
            other_possible_actions = list(set(self.possible_actions) \
                                          - set([child.action]))

            # find some random index in that list
            rand_idx = int(np.floor(np.random.uniform() * \
                                    len(other_possible_actions)))

            # assign the action of this child to that random action
            child.action = other_possible_actions[rand_idx]

    def does_subsume(self, cl_sub, cl_tos):
        # if cl_sub and cl_tos have the same action
        if cl_sub.action == cl_tos.action:
            # if cl_sub is allowed to subsume another classifier
            if self.could_subsume(cl_sub):
                # is cl_sub is more general than cl_tos
                # if self.is_more_general(cl_sub, cl_tos):
                # 	# then cl_sub does subsume cl_tos
                return True

        # otherwise, cl_sub does not subsume cl_tos
        return False

    def is_more_general(self, cl_gen, cl_spec):
        # count the number of wildcards in cl_gen
        cl_gen_wildcard_count = cl_gen.count_wildcards()

        # count the number of wildcards in cl_spec
        cl_spec_wildcard_count = cl_spec.count_wildcards()

        # if cl_gen is not more general than cl_spec
        if cl_gen_wildcard_count <= cl_spec_wildcard_count:
            return False

        # for each attribute index i in the classifiers condition
        for i in range(self.condition_length):
            # if the condition for cl_gen is not the wildcard
            # and cl_gen condition[i] does not match cl_spec condition[i]
            if cl_gen.condition[i] != Classifier.WILDCARD_ATTRIBUTE_VALUE and \
                    cl_gen.condition[i] != cl_spec.condition[i]:
                # then cl_gen is not more general than cl_spec
                return False

        # otherwise, cl_gen is more general than cl_spec
        return True

    def could_subsume(self, classifier):
        # if the classifier's experience is greater than
        # the subsumption threshold
        if classifier.experience > self.theta_sub:
            # and if the classifier's error (epsilon) is less than
            # the error threashold
            if classifier.epsilon < self.epsilon_0:
                # return true i.e. this classifier can subsume another
                return True

        # othwerise, this classifier cannot subsume another
        return False

    def do_action_set_subsumption_procedure(self, set_):
        # initialize an empty classifier
        cl = None

        # create a local variable to represent the number of
        # wildcards that appear in the classifier cl's condition
        cl_wildcard_count = 0

        # for each classifier in the set_
        for c in set_:
            # if c is able to subsume other classifiers
            if self.could_subsume(c):
                # if cl is empty or the number of wildcards in c is greater
                # than the number of wildcards in cl or the number of
                # wildcards in c equals the number of wildcards in cl and
                # some random value is less than 0.5
                if cl is None or \
                        c.count_wildcards() > cl_wildcard_count or \
                        (cl_wildcard_count == c.count_wildcards() and \
                         np.random.uniform() < 0.5):
                    # then set cl to c
                    cl = c

                    # update the cl_wildcard_count to equal the wildcard
                    # cound in c
                    cl_wildcard_count = c.count_wildcards()

        # if cl is not empty
        if cl is not None:
            # for each classifier in the set_
            for c in set_:
                # if cl is more general than c, then subsume it
                if self.is_more_general(cl, c):
                    # increlemtn cl's numberosity
                    cl.numerosity += c.numerosity

                    # remove c from the set_
                    set_.remove(c)

                    # if c is in the population
                    if c in self.population:
                        # then remove it from the population
                        self.population.remove(c)

    def delete_from_population(self):
        # get the total number of micro-classifiers
        # currently present in the population
        num_micro_classifiers = sum([cl.numerosity for cl in self.population])

        # if the number of classifiers is less than the max allowed
        if num_micro_classifiers <= self.N:
            # return i.e. do nothing
            return

        # the the total population for all the classifiers
        # currently present in the population
        sum_population_fitness = sum([cl.fitness for cl in self.population])

        # compute the average fitness over all the classifiers
        # currently present in the population
        avg_fitness_in_population = sum_population_fitness / \
                                    num_micro_classifiers

        # set a local variable to track the deletion vote of all the classifiers
        vote_sum = 0

        # for each classifier currently in the population
        for cl in self.population:
            # sum the deletion vote of all the classifiers
            vote_sum += self.deletion_vote(cl, avg_fitness_in_population)

        # select a random threashold for vote_sum
        choice_point = np.random.uniform() * vote_sum

        # reset the vote_sum to zero
        vote_sum = 0

        # for each classifier currently in the population
        for cl in self.population:
            # sum the deletion vote of all the classifiers
            vote_sum += self.deletion_vote(cl, avg_fitness_in_population)

            # if the current vote_sum is larger than our random threashold
            if vote_sum > choice_point:
                # if the numerosity of this classifier is > 1
                if cl.numerosity > 1:
                    # decrement its numerosity
                    cl.numerosity -= 1
                else:
                    # otherwise, if its 1, remove it from the population
                    self.population.remove(cl)

                return

    def deletion_vote(self, classifier, avg_fitness_in_population):
        # compute the vote-value for this classifier
        vote = classifier.action_set_size * classifier.numerosity

        # compute the weighted fitness of this classifier
        # accounting for the classifier's numerosity
        fitness_per_numerosity = classifier.fitness / classifier.numerosity

        # if this classifier's experience > the deletion threashold
        # and fitness_per_numerosity < the fration
        # of the mean fitness in population * the average fitness
        if classifier.experience > self.theta_del and \
                fitness_per_numerosity < \
                (self.delta * avg_fitness_in_population):
            # set the vote to vote * average fitness / fitness_per_numerosity
            vote = (vote * avg_fitness_in_population) / fitness_per_numerosity

        return vote

    def insert_in_population(self, classifier):
        # for each classifier currently in the population
        for cl in self.population:
            # if the other classifier is equal to the parameter
            # classifier in both condition and action
            if cl.condition == classifier.condition and \
                    cl.action == classifier.action:
                # then increment the other classifier's numerosity
                cl.numerosity += 1

                # and exit, i.e. dont add the parameter classifier
                # to the population
                return

        # if this classifier is unique then add it to the population
        self.population.append(classifier)
