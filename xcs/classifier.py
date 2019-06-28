# Brodderick Rodriguez
# Auburn University - CSSE
# June 6 2019


class Classifier:
	WILDCARD_ATTRIBUTE_VALUE = '#'
	POSSIBLE_ATTRIBUTE_VALUES = [0, 1, WILDCARD_ATTRIBUTE_VALUE]
	CLASSIFIER_ID = 0

	# xcs_object: the XCS we are using. used to retrieve default values
	def __init__(self, xcs_object):
		self.xcs_object = xcs_object

		self.id = Classifier.CLASSIFIER_ID
		Classifier.CLASSIFIER_ID += 1

		# condition that specifies the sensory situation
		# which the classifier applies to
		self.condition = [0 for _ in range(xcs_object.condition_length)]

		# action the classifier proposes
		self.action = None

		# (p) estimated payoff expected if the classifier matches and
		# its action is committed
		self.predicted_payoff = xcs_object.p_1

		# (epsilon) the error made in the predictions
		self.epsilon = xcs_object.epsilon_1

		# (F) the classifiers fitness
		self.fitness = xcs_object.F_1

		# (exp) count for the number of times this classifier has
		# belonged to the action_set
		self.experience = 0

		# time_step of the last occurrence of a GA in an
		# action_set to which this classifier belonged
		self.last_time_step = 0

		# (as) average size of the action_set this classifier
		# belongs to
		self.action_set_size = 0

		# number of micro-classifiers this classifier represents
		self.numerosity = 1

	def __str__(self):
		s = '\nid: {id}\n\tcondition: {cond}, action: {act}\n\tpred:\t{pred} \
				\n\terror:\t{err}\n\tfit:\t{fit} \
				\n\tnum:\t{num}\n\texp:\t{exp}\n\t'.format(
					id=self.id, cond=self.condition, act=self.action,
					pred=self.predicted_payoff, err=self.epsilon,
					fit=self.fitness, num=self.numerosity, exp=self.experience)
		return s

	def __repr__(self):
		return self.__str__()

	def copy(self):
		other = Classifier(self.xcs_object)
		other.__dict__ = self.__dict__
		return other

	def count_wildcards(self):
		count = sum([1 if x == Classifier.WILDCARD_ATTRIBUTE_VALUE else 0 for x in self.condition])
		return count
