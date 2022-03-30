#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
A class that keeps track of the accuracy metric during training on each
dataset and logs the predictions and labels into a .csv file.
"""

import csv
import numpy as np

from aux import get_predictions_path


class EpochEvaluator:
    """Log data over one epoch."""
    def __init__(self):
        self.total_loss = 0
        self.indexes = []
        self.predictions = []
        self.labels = []

    def update_loss(self, loss):
        """Add a new batch loss."""
        self.total_loss += loss

    def update_datapoints(self, item_ids, labels):
        """Add a new batch (indexes and their labels)."""
        self.indexes += item_ids
        self.labels += labels

    def update_predictions(self, predicted):
        "Add a new batch of predictions."
        self.predictions += predicted

    @property
    def acc(self):
        """Compute accuracy over all predictions."""
        assert len(self.predictions) == len(self.labels)
        right = np.sum(np.array(self.predictions) == np.array(self.labels))
        return right / len(self.predictions)

    def print_state(self):
        """Print current loss and accuracy."""
        print(' \t loss: {:.2f}'.format(self.total_loss))
        print(' \t acc: {:.2f}'.format(self.acc))

    def log_predictions(self, name, mode, epoch):
        """Save all predictions to a .csv. file."""
        out_path = get_predictions_path(name, mode, epoch)
        with open(out_path, 'w') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(['global_id', 'prediction', 'gold label'])
            for ind, pred, gold in zip(self.indexes,
                                       self.predictions,
                                       self.labels):
                csv_out.writerow([int(ind), int(pred), int(gold)])


class Evaluator():
    """An evaluator to log and store accuracy during learning."""
    def __init__(self, name, early_stop):
        """
        Args:
            name (str): The directory name where data is to be logged.
            early_stop (int): Number of iterations needed to do early stop.
        """
        self.name = name
        self.early_stop = early_stop
        self.performance = {'train': [], 'valid': [], 'test': []}
        self.loss = {'train': [], 'valid': [], 'test': []}
        # initialize variables for early stopping and to retrieve best model
        self.best_valid_performance = -np.inf
        self.best_valid_epoch = -np.inf

    def eval(self, mode, epoch, eval_epoch):
        """Log metrics information and predictions after a complete epoch.

        Args:
            mode (str): Train, valid or test.
            epoch (int): Number of current epoch.
            eval_epoch (EpochEvaluator): an EpochEvaluator populated after one
                                         complete epoch.

        Returns:
            float: Accuracy of current predicted labels.
        """
        mode = mode.replace('init_', '')
        acc = eval_epoch.acc
        self.performance[mode].append(acc)
        self.loss[mode].append(eval_epoch.total_loss)
        eval_epoch.print_state()
        if mode != 'train':
            eval_epoch.log_predictions(self.name, mode, epoch)
        return acc

    def make_early_stop(self, epoch):
        """Return True if early stopping criteria is met."""
        return self.best_valid_epoch <= (epoch - self.early_stop)

    def update_best(self, epoch):
        """Update state of best epoch re. performance on valid set."""
        if self.performance['valid'][-1] < self.best_valid_performance:
            return False
        self.best_valid_performance = self.performance['valid'][-1]
        self.best_valid_epoch = epoch
        return True
