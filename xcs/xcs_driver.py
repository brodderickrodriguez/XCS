# Brodderick Rodriguez
# Auburn University - CSSE
# June 28 2019

from xcs.xcs import XCS

import numpy as np
import multiprocessing
import logging
import os
import time


class XCSDriver:
    def __init__(self):
        print('driver initialized')

        self.xcs_class = None
        self.reinforcement_program_class = None
        self.environment_class = None
        self.configuration_class = None

        self.repetitions = 10

        self.save_location = './'
        self.experiment_name = None

        self._root_data_directory = None

        self.data = {'rhos': [], 'predicted_rhos': [], 'microclassifier_counts': []}

    def run(self):
        logging.info('Running XCSDriver')

        self._check_arguments()
        logging.info('XCSDriver passed argument check')

        self._setup_directories()
        logging.info('XCSDriver created directory: {}' \
                     .format(self._root_data_directory))

        self._run_processes()
        logging.info('XCSDriver ran all processes')

        return self.data

    def _check_arguments(self):
        pass

    def _setup_directories(self):
        self.experiment_name = self.experiment_name or str(int(time.time()))

        self._root_data_directory = self.save_location + '/' + self.experiment_name

        print(os.path.abspath('./'))

        directories = ['', '/classifiers', '/results', '/results/rhos', '/results/predicted_rhos',
                       '/results/microclassifier_counts']

        for directory in directories:
            os.mkdir(self._root_data_directory + directory)

    def _run_processes(self):
        for i in range(20):
            self._run_repetition(i)

    def _run_repetition(self, repetition_num):
        config = self.configuration_class()
        env = self.environment_class()
        rp = self.reinforcement_program_class(configuration=config)

        xcs_object = XCS(environment=env, reinforcement_program=rp, configuration=config)
        xcs_object.run_experiment()

        self._save_repetition(xcs_object.metrics_history, repetition_num)

    def _save_repetition(self, metrics, repetition_num):
        for key, val in metrics.items():
            self.data[key].append(val)

        # the path to where results are stored
        path = self._root_data_directory + '/results/'

        for key in metrics.keys():
            # the filename where we will store this metric
            filename = path + key + '/repetition' + str(repetition_num) + '.csv'

            # if the metric does not have length, i.e. its a scalar
            # then we handle it differently
            if not hasattr(metrics[key][0], '__len__'):
                data = np.array(metrics[key])
            else:
                # we can have n-D arrays of variable length
                # here we find the array with the longest length
                M = max([len(e) for e in metrics[key]])

                # then we shape our data into and NxM matrix
                data = np.zeros((len(metrics[key]), M))

                # set all cells to nan
                data[:] = np.nan

                # iterate over each column of the metric
                for i in range(len(metrics[key])):
                    # iterate over each cell fo the metric
                    for j in range(len(metrics[key][i])):
                        # save that value to its corresponding cell in data
                        data[i, j] = metrics[key][i][j]

            # finally, save our NxM matrix
            np.savetxt(filename, data, delimiter=',')

        print('repetition {} done'.format(repetition_num))
