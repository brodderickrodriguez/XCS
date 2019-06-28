# Brodderick Rodriguez
# Auburn University - CSSE
# June 11 2019

import numpy as np


def print_best_classifiers(xcs_object, n=5):
	sorted_ = sorted(xcs_object.population,
						key=lambda x: x.predicted_payoff, 
						reverse=True)
	n_best = sorted_[:n]
	print('\n{} best classifers in population:\n{}'.format(n, n_best))


def print_mean_reward(xcs_object, time_frame=100):
	reward = xcs_object.metrics_history['rhos'][-time_frame:]
	mean_reward = np.mean(reward)
	print('mean reward over {tf} step time frame is: {mr}'.format(tf=time_frame, 
															mr=mean_reward))


def print_reward_metrics(xcs_object, time_frame=0):
	reward = xcs_object.rp.results[:]

	correct = sum([r for r in reward])
	incorrect = len(reward) - correct

	print('correct:\t{}\nincorrect:\t{}\ntotal:\t\t{}\npercentage:\t{}'.
		format(correct, incorrect, len(reward), float(correct/len(reward))))
