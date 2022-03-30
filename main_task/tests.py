#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Unit testing for the main scripts.
"""

import unittest

from collections import Counter

import aux
import config
import dataloader
import evaluate


class TestAuxFunctions(unittest.TestCase):
    """Test functions in the aux.py."""
    def test_counter_to_percentage(self):
        counter = Counter([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        percentages = aux.counter_to_percentage(counter)
        self.assertEqual(percentages[1], 0.1)
        self.assertEqual(percentages[2], 0.2)
        self.assertEqual(percentages[3], 0.3)
        self.assertEqual(percentages[4], 0.4)


class TestEvaluator(unittest.TestCase):
    """Test class methods of main evaluator in evaluate.py."""
    def test_early_stop(self):
        evaluator = evaluate.Evaluator('test', early_stop=5)

        evaluator.best_valid_epoch = 0
        self.assertFalse(evaluator.make_early_stop(epoch=0))
        self.assertFalse(evaluator.make_early_stop(epoch=4))
        self.assertTrue(evaluator.make_early_stop(epoch=5))
        self.assertTrue(evaluator.make_early_stop(epoch=1000))

        evaluator.best_valid_epoch = 10
        self.assertFalse(evaluator.make_early_stop(epoch=11))
        self.assertFalse(evaluator.make_early_stop(epoch=14))
        self.assertTrue(evaluator.make_early_stop(epoch=15))
        self.assertTrue(evaluator.make_early_stop(epoch=20))

    def test_update_best(self):
        evaluator = evaluate.Evaluator('test', early_stop=5)

        evaluator.performance['valid'].append(0.6)
        self.assertTrue(evaluator.update_best(epoch=0))

        evaluator.best_valid_epoch = 2
        evaluator.best_valid_performance = 0.6
        evaluator.performance['valid'] = [0.4, 0.5, 0.6, 0.5]
        update = evaluator.update_best(epoch=3)
        self.assertFalse(update)
        self.assertEqual(evaluator.best_valid_performance, 0.6)
        self.assertEqual(evaluator.best_valid_epoch, 2)

        evaluator.best_valid_epoch = 2
        evaluator.best_valid_performance = 0.6
        evaluator.performance['valid'] = [0.4, 0.5, 0.6, 0.7]
        update = evaluator.update_best(epoch=3)
        self.assertTrue(update)
        self.assertEqual(evaluator.best_valid_performance, 0.7)
        self.assertEqual(evaluator.best_valid_epoch, 3)


class TestEpochEvaluator(unittest.TestCase):
    """Test class methods of epoch evaluator in evaluate.py."""
    def test_update_loss(self):
        epoch_eval = evaluate.EpochEvaluator()

        epoch_eval.update_loss(10)
        self.assertEqual(epoch_eval.total_loss, 10)
        epoch_eval.update_loss(200)
        self.assertEqual(epoch_eval.total_loss, 210)

    def test_update_datapoints(self):
        epoch_eval = evaluate.EpochEvaluator()
        indexes_1 = [234, 656, 342]
        labels_1 = [1, 2, 0]
        epoch_eval.update_datapoints(indexes_1, labels_1)
        self.assertEqual(epoch_eval.indexes, indexes_1)
        self.assertEqual(epoch_eval.labels, labels_1)

        indexes_2 = [82, 196]
        labels_2 = [2, 0]
        epoch_eval.update_datapoints(indexes_2, labels_2)
        self.assertEqual(epoch_eval.indexes, indexes_1 + indexes_2)
        self.assertEqual(epoch_eval.labels, labels_1 + labels_2)

    def test_update_predictions(self):
        epoch_eval = evaluate.EpochEvaluator()
        predictions_1 = [1, 0, 2]
        epoch_eval.update_predictions(predictions_1)
        self.assertEqual(epoch_eval.predictions, predictions_1)

        predictions_2 = [1, 0, 1, 2]
        epoch_eval.update_predictions(predictions_2)
        self.assertEqual(epoch_eval.predictions, predictions_1 + predictions_2)

    def test_acc(self):
        epoch_eval = evaluate.EpochEvaluator()
        predictions = [1, 1, 2, 0, 0, 2, 1, 0, 0, 2]
        labels = [2, 1, 2, 1, 0, 2, 2, 0, 0, 1]
        idxs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        epoch_eval.update_datapoints(idxs, labels)
        epoch_eval.update_predictions(predictions)
        acc = epoch_eval.acc
        self.assertEqual(acc, 0.6)


class TestProbingDataset(unittest.TestCase):
    """Test class methods of dataloader.py."""

    def setUp(self):
        self.params = config.args()
        self.params.task = 'TFxPS'
        self.data_tfxps = dataloader.ProbingDataset(self.params, 'train')
        self.params.task = 'TF'
        self.data_tf = dataloader.ProbingDataset(self.params, 'train')
        self.params.task = 'PS'
        self.data_ps = dataloader.ProbingDataset(self.params, 'train')
        self.params.task = 'PxTSFS'
        self.data_ptsfs = dataloader.ProbingDataset(self.params, 'train')

    def test_create_labels(self):
        # TFxPS
        #  a true proposition that becomes shared at turn 5
        labels = self.data_tfxps._create_labels(1, 5, 11)
        correct_labels = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]
        self.assertEqual(labels, correct_labels)
        #  a false proposition that becomes shared at turn 1
        labels = self.data_tfxps._create_labels(0, 1, 11)
        correct_labels = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertEqual(labels, correct_labels)
        #  a true proposition that becomes shared at turn 0, shorter dialogue
        labels = self.data_tfxps._create_labels(1, 0, 3)
        correct_labels = [0, 0, 0]
        self.assertEqual(labels, correct_labels)
        #  a false proposition that becomes shared at turn 6, shorter dialogue
        labels = self.data_tfxps._create_labels(0, 6, 7)
        correct_labels = [3, 3, 3, 3, 3, 3, 1]
        self.assertEqual(labels, correct_labels)

        # TF
        #  a true proposition that becomes shared at turn 5
        labels = self.data_tf._create_labels(1, 5, 11)
        correct_labels = [0] * 11
        self.assertEqual(labels, correct_labels)
        #  a false proposition that becomes shared at turn 1
        labels = self.data_tf._create_labels(0, 1, 11)
        correct_labels = [1] * 11
        self.assertEqual(labels, correct_labels)
        #  a true proposition that becomes shared at turn 0, shorter dialogue
        labels = self.data_tf._create_labels(1, 0, 3)
        correct_labels = [0] * 3
        self.assertEqual(labels, correct_labels)
        #  a false proposition that becomes shared at turn 6, shorter dialogue
        labels = self.data_tf._create_labels(0, 6, 7)
        correct_labels = [1] * 7
        self.assertEqual(labels, correct_labels)

        # PS
        #  a true proposition that becomes shared at turn 5
        labels = self.data_ps._create_labels(1, 5, 11)
        correct_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        self.assertEqual(labels, correct_labels)
        #  a false proposition that becomes shared at turn 1
        labels = self.data_ps._create_labels(0, 1, 11)
        correct_labels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(labels, correct_labels)
        #  a true proposition that becomes shared at turn 0, shorter dialogue
        labels = self.data_ps._create_labels(1, 0, 3)
        correct_labels = [0, 0, 0]
        self.assertEqual(labels, correct_labels)
        #  a false proposition that becomes shared at turn 6, shorter dialogue
        labels = self.data_ps._create_labels(0, 6, 7)
        correct_labels = [1, 1, 1, 1, 1, 1, 0]
        self.assertEqual(labels, correct_labels)      

        # PxTSFS
        #  a true proposition that becomes shared at turn 5
        labels = self.data_ptsfs._create_labels(1, 5, 11)
        correct_labels = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]
        self.assertEqual(labels, correct_labels)
        #  a false proposition that becomes shared at turn 1
        labels = self.data_ptsfs._create_labels(0, 1, 11)
        correct_labels = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertEqual(labels, correct_labels)
        #  a true proposition that becomes shared at turn 0, shorter dialogue
        labels = self.data_ptsfs._create_labels(1, 0, 3)
        correct_labels = [0, 0, 0]
        self.assertEqual(labels, correct_labels)
        #  a false proposition that becomes shared at turn 6, shorter dialogue
        labels = self.data_ptsfs._create_labels(0, 6, 7)
        correct_labels = [2, 2, 2, 2, 2, 2, 1]
        self.assertEqual(labels, correct_labels)

    def test_create_new_items(self):
        d_idx = 134000
        p_idx = 10
        s_idx = 562
        idxs = (d_idx, p_idx, s_idx)
        labels = self.data_ptsfs._create_labels(1, 4, 6)
        new_items = self.data_tfxps._create_new_items(6, idxs, labels)
        new_items = {x[0]: x[1:] for x in new_items}
        size = len(self.data_tfxps.datapoints)
        real_items = {size + 0: (d_idx, p_idx, s_idx, 0, 2),
                      size + 1: (d_idx, p_idx, s_idx, 1, 2),
                      size + 2: (d_idx, p_idx, s_idx, 2, 2),
                      size + 3: (d_idx, p_idx, s_idx, 3, 2),
                      size + 4: (d_idx, p_idx, s_idx, 4, 0),
                      size + 5: (d_idx, p_idx, s_idx, 5, 0),
                      }
        self.assertDictEqual(new_items, real_items)

    def test_varying_test_lengths(self):
        data_tfxps = dataloader.ProbingDataset(self.params, 'test')
        path = aux.get_test_lens_path()
        with open(path, 'r') as f:
            lens = {}
            for line in f.readlines():
                idx, length = line.strip('\n').split('\t')
                lens[int(idx)] = int(length)
        for d_id, p_id, sent_id, turn, label in data_tfxps.datapoints.values():
            self.assertLess(turn, lens[d_id])


if __name__ == '__main__':
    unittest.main()
