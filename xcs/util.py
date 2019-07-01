# Brodderick Rodriguez
# Auburn University - CSSE
# July 1 2019

import os
import numpy as np
import matplotlib.pyplot as plt


def load_metric(experiment_path, metric):
    results_path = experiment_path + '/results/' + metric
    contents = os.listdir(results_path)

    first_repetition = np.loadtxt(results_path + '/' + contents[0])

    data = np.zeros((len(contents), len(first_repetition)))
    data[0] = first_repetition
    data[data == 0] = np.nan

    for i, file in enumerate(contents[1:]):
        data[i + 1] = np.loadtxt(results_path + '/' + file)

    return data


def load_results(experiment_path):
    rhos = load_metric(experiment_path, 'rhos')
    predicted_rhos = load_metric(experiment_path, 'predicted_rhos')
    classifier_counts = load_metric(experiment_path, 'microclassifier_counts')
    return rhos, predicted_rhos, classifier_counts


def plot_results(experiment_path, interval=50, title=''):
    rhos, pred_rhos, cc = load_results(experiment_path)
    error = np.abs(rhos - pred_rhos)

    rhos_means = np.nanmean(rhos, axis=0)
    error_means = np.nanmean(error, axis=0)
    cc_means = np.nanmean(cc, axis=0)

    x, rho_y, error_y, cc_y = [], [], [], []

    for xi in range(interval, len(rhos_means), interval):
        rho_yi = np.mean(rhos_means[xi - interval: xi])
        error_yi = np.mean(error_means[xi - interval: xi])
        cc_yi = np.mean(cc_means[xi - interval: xi])

        x.append(xi / 1000)
        rho_y.append(rho_yi)
        error_y.append(error_yi)
        cc_y.append(cc_yi / 1000)

    plt.xlabel('episodes (thousands)')
    plt.plot(x, rho_y, label='reward')
    plt.plot(x, error_y, label='error', linestyle='--')
    plt.plot(x, cc_y, label='Pop. size (/1000)')
    plt.title(title)
    plt.gca().legend()
    plt.show()
