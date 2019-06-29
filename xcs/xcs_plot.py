# Brodderick Rodriguez
# Auburn University - CSSE
# June 19 2019

import numpy as np
import matplotlib.pyplot as plt


def plot(xcs_object, interval=50, title=''):

	rhos = np.array(xcs_object.metrics_history['rhos'])
	pred_rhos = xcs_object.metrics_history['predicted_rhos']

	reward_means = np.mean(rhos, axis=0)
	error_means = np.mean(np.abs(rhos - pred_rhos), axis = 0)

	x, reward_y, error_y = [], [], []

	for xi in range(interval, len(reward_means), interval):
		reward_yi = np.mean(reward_means[xi - interval: xi])
		error_yi = np.mean(error_means[xi - interval: xi])

		x.append(xi / 1000)
		reward_y.append(reward_yi)
		error_y.append(error_yi)

	plt.xlabel('episodes (thousands)')

	plt.plot(x, reward_y, label='reward')
	plt.plot(x, error_y, label='error', linestyle='--')
	plt.title(title)
	plt.gca().legend()
	plt.show()


def plot2(data, interval=50, title=''):

	rhos = np.array(data['rhos'])
	pred_rhos = data['predicted_rhos']

	reward_means = np.mean(rhos, axis=0)
	error_means = np.mean(np.abs(rhos - pred_rhos), axis = 0)

	x, reward_y, error_y = [], [], []

	for xi in range(interval, len(reward_means), interval):
		reward_yi = np.mean(reward_means[xi - interval: xi])
		error_yi = np.mean(error_means[xi - interval: xi])

		x.append(xi / 1000)
		reward_y.append(reward_yi)
		error_y.append(error_yi)

	plt.xlabel('episodes (thousands)')

	plt.plot(x, reward_y, label='reward')
	plt.plot(x, error_y, label='error', linestyle='--')
	plt.title(title)
	plt.gca().legend()
	plt.show()
