#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Unit testing for evaluation scripts.
"""

from ast import Assert
import unittest
import numpy as np
from numpy import testing as npt

import evaluationaux as ev
from evaluationtypes import Proposition, Dialogue
from tasks import get_task_labels


class TestProposition(unittest.TestCase):
    """Test methods in Proposition class."""
    def setUp(self):
        p = {'proposition': 'there is no dog.', 'a_thinks_true': 1,
              'turn_shared': 4, 'rule': 'is_it', 'qa_fact': 'negative'}
        _, TFxPS_labels = get_task_labels('a', 'TFxPS')
        self.probe_1 = Proposition(0, 10, p, TFxPS_labels, 'a', 'TFxPS')

        p = {'proposition': 'there is no dog.', 'a_thinks_true': 1,
              'turn_shared': 0, 'rule': 'is_it', 'qa_fact': 'negative'}
        _, TF_labels = get_task_labels('a', 'TF')
        self.probe_2 = Proposition(0, 10, p, TF_labels, 'a', 'TF')

        p = {'proposition': 'there is no dog.', 'a_thinks_true': 1,
              'turn_shared': 11, 'rule': 'is_it', 'qa_fact': 'negative'}
        _, PS_labels = get_task_labels('a', 'PS')
        self.probe_3 = Proposition(0, 10, p, PS_labels, 'a', 'PS')

        p = {'proposition': 'there is no dog.', 'a_thinks_true': 1,
              'turn_shared': 7, 'rule': 'is_it', 'qa_fact': 'negative'}
        _, PxTSFS_labels = get_task_labels('a', 'PxTSFS')
        self.probe_4 = Proposition(0, 10, p, PxTSFS_labels, 'a', 'PxSTSF')

    def test_populate(self):
        """Populating assigns prediction and gold to correct turn."""
        # all start as inf
        preds = np.full(11, np.inf)
        gold = np.full(11, np.inf)
        np.testing.assert_array_equal(self.probe_1.predictions, preds)
        np.testing.assert_array_equal(self.probe_1.gold, gold)
        # make a few assignments
        self.probe_1.populate(3, 4, 5)
        self.probe_1.populate(2, 2, 0)
        self.probe_1.populate(1, 1, 10)
        preds = np.array([2, np.inf, np.inf, np.inf, np.inf, 4, np.inf,
                          np.inf, np.inf, np.inf, 1])
        gold = np.array([2, np.inf, np.inf, np.inf, np.inf, 3, np.inf,
                          np.inf, np.inf, np.inf, 1])
        np.testing.assert_array_equal(self.probe_1.predictions, preds)
        np.testing.assert_array_equal(self.probe_1.gold, gold)

    def test_is_fully_populated(self):
        """Return True only when all turns have predictions and labels."""
        # should be false until all turns were populated
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 0)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 1)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 2)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 3)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 4)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 5)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 6)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 7)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 8)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 9)
        self.assertFalse(self.probe_1.is_fully_populated)
        self.probe_1.populate(3, 4, 10)
        self.assertTrue(self.probe_1.is_fully_populated)

    def test_acc(self):
        """Compute accuracy."""
        self.probe_1.predictions = np.array([1, 1, 2, 3, 3, 2, 1, 1, 2, 3, 1])
        self.probe_1.gold = np.array([1, 2, 2, 1, 3, 1, 1, 1, 3, 3, 1])
        acc = 7 / 11
        self.assertEqual(self.probe_1.acc, acc)

    def test_tf_acc(self):
        """Computer accuracy on TF dimension only."""
        # TFxPS
        self.probe_1.predictions = np.array([0, 1, 2, 3, 3, 1, 1, 0, 2, 3, 1])
        self.probe_1.gold =        np.array([0, 2, 3, 0, 0, 3, 2, 1, 1, 2, 3])
        tf_acc = 3 / 11
        self.assertEqual(self.probe_1.tf_acc, tf_acc)
        self.probe_1.predictions = np.array([0, 2, 1, 2, 2, 1, 1, 0, 2, 3, 2])
        self.probe_1.gold =        np.array([0, 2, 3, 0, 0, 3, 2, 1, 1, 2, 3])
        tf_acc = 6 / 11
        self.assertEqual(self.probe_1.tf_acc, tf_acc)
        # TF
        self.probe_2.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.probe_2.gold =        np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        tf_acc = 9 / 11
        self.assertEqual(self.probe_2.tf_acc, tf_acc)
        self.probe_2.predictions = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1])
        self.probe_2.gold =        np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        tf_acc = 7 / 11
        self.assertEqual(self.probe_2.tf_acc, tf_acc)
        # PS
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.probe_3.gold =        np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        with self.assertRaises(AssertionError):
            _ = self.probe_3.tf_acc
        # PxSTSF
        self.probe_4.predictions = np.array([0, 1, 2, 0, 1, 2, 2, 0, 1, 0, 1])
        self.probe_4.gold =        np.array([0, 2, 1, 0, 2, 2, 0, 1, 1, 0, 1])
        with self.assertRaises(AssertionError):
            _ = self.probe_4.tf_acc

    def test_ps_acc(self):
        """Compute accuracy on PS dimension only."""
        # TFxPS
        self.probe_1.predictions = np.array([0, 1, 2, 3, 3, 1, 1, 0, 2, 3, 1])
        self.probe_1.gold =        np.array([0, 2, 3, 0, 0, 3, 2, 1, 1, 2, 3])
        ps_acc = 4 / 11
        self.assertEqual(self.probe_1.ps_acc, ps_acc)
        self.probe_1.predictions = np.array([0, 2, 1, 2, 2, 1, 1, 0, 2, 3, 2])
        self.probe_1.gold =        np.array([0, 2, 3, 0, 0, 3, 2, 1, 1, 2, 3])
        ps_acc = 5 / 11
        self.assertEqual(self.probe_1.ps_acc, ps_acc)
        # TF
        self.probe_2.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.probe_2.gold =        np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        with self.assertRaises(AssertionError):
            _ = self.probe_2.ps_acc
        # PS
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.probe_3.gold =        np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        ps_acc = 9 / 11
        self.assertEqual(self.probe_3.ps_acc, ps_acc)
        self.probe_3.predictions = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1])
        self.probe_3.gold =        np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        ps_acc = 7 / 11
        self.assertEqual(self.probe_3.ps_acc, ps_acc)
        # PxSTSF
        self.probe_4.predictions = np.array([0, 1, 2, 0, 1, 2, 2, 0, 1, 0, 1])
        self.probe_4.gold =        np.array([0, 2, 1, 0, 2, 2, 0, 1, 1, 0, 1])
        ps_acc = 7 / 11
        self.assertEqual(self.probe_4.ps_acc, ps_acc)
        self.probe_4.predictions = np.array([0, 1, 0, 0, 2, 2, 2, 0, 0, 1, 1])
        self.probe_4.gold =        np.array([0, 2, 1, 0, 2, 2, 0, 1, 1, 0, 1])
        ps_acc = 9 / 11
        self.assertEqual(self.probe_4.ps_acc, ps_acc)

    def test_n_labels(self):
        """Count size of set of predictions."""
        self.probe_1.predictions = np.array([0, 1, 2, 3, 3, 1, 1, 0, 2, 3, 1])
        self.assertEqual(self.probe_1.n_labels, 4)
        self.probe_1.predictions = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1])
        self.assertEqual(self.probe_1.n_labels, 2)
        self.probe_1.predictions = np.array([0, 1, 2, 1, 0, 1, 2, 2, 1, 0, 1])
        self.assertEqual(self.probe_1.n_labels, 3)

    def test_n_shifts(self):
        """Compute how many shifts were predicted."""
        self.probe_1.predictions = np.array([0, 1, 2, 3, 3, 1, 1, 0, 2, 3, 1])
        self.assertEqual(self.probe_1.n_shifts, 8)
        self.probe_1.predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(self.probe_1.n_shifts, 0)
        self.probe_1.predictions = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        self.assertEqual(self.probe_1.n_shifts, 5)
        self.probe_1.predictions = np.array([1, 1, 2, 3, 3, 1, 0, 0, 0, 3, 1])
        self.assertEqual(self.probe_1.n_shifts, 6)

    def test_n_tf_shifts(self):
        """Compute how many shifts on TF dimension only."""
        # TFxPS
        self.probe_1.predictions = np.array([0, 1, 2, 3, 3, 1, 1, 0, 2, 3, 1])
        self.assertEqual(self.probe_1.n_tf_shifts, 5)
        self.probe_1.predictions = np.array([0, 2, 1, 1, 3, 1, 3, 0, 2, 0, 1])
        self.assertEqual(self.probe_1.n_tf_shifts, 3)
        self.probe_1.predictions = np.array([0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2])
        self.assertEqual(self.probe_1.n_tf_shifts, 0)
        self.probe_1.predictions = np.array([1, 3, 3, 1, 1, 3, 3, 3, 1, 1, 3])
        self.assertEqual(self.probe_1.n_tf_shifts, 0)
        # TF
        self.probe_2.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.assertEqual(self.probe_2.n_tf_shifts, 7)
        self.probe_2.predictions = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1])
        self.assertEqual(self.probe_2.n_tf_shifts, 3)
        self.probe_2.predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(self.probe_2.n_tf_shifts, 0)
        self.probe_2.predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(self.probe_2.n_tf_shifts, 0)
        # PS
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        with self.assertRaises(AssertionError):
            _ = self.probe_3.n_tf_shifts
        # PxSTSF
        self.probe_4.predictions = np.array([0, 1, 2, 0, 1, 2, 2, 0, 1, 0, 1])
        with self.assertRaises(AssertionError):
            _ = self.probe_4.n_tf_shifts

    def test_n_ps_shifts(self):
        """Compute how many shifts on PS dimension only."""
        # TFxPS
        self.probe_1.predictions = np.array([0, 1, 2, 3, 3, 1, 1, 0, 2, 3, 1])
        self.assertEqual(self.probe_1.n_ps_shifts, 4)
        self.probe_1.predictions = np.array([0, 2, 1, 1, 3, 1, 3, 0, 2, 0, 1])
        self.assertEqual(self.probe_1.n_ps_shifts, 8)
        self.probe_1.predictions = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1])
        self.assertEqual(self.probe_1.n_ps_shifts, 0)
        self.probe_1.predictions = np.array([3, 2, 3, 2, 3, 3, 3, 2, 2, 3, 2])
        self.assertEqual(self.probe_1.n_ps_shifts, 0)
        # TF
        self.probe_2.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        with self.assertRaises(AssertionError):
            _ = self.probe_2.n_ps_shifts
        # PS
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.assertEqual(self.probe_3.n_ps_shifts, 7)
        self.probe_3.predictions = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1])
        self.assertEqual(self.probe_3.n_ps_shifts, 3)
        self.probe_3.predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(self.probe_3.n_ps_shifts, 0)
        self.probe_3.predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(self.probe_3.n_ps_shifts, 0)
        # PxSTSF
        self.probe_4.predictions = np.array([0, 1, 0, 0, 2, 2, 2, 0, 0, 1, 1])
        self.assertEqual(self.probe_4.n_ps_shifts, 2)
        self.probe_4.predictions = np.array([0, 2, 0, 1, 2, 1, 2, 0, 2, 1, 1])
        self.assertEqual(self.probe_4.n_ps_shifts, 8)
        self.probe_4.predictions = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1])
        self.assertEqual(self.probe_4.n_ps_shifts, 0)
        self.probe_4.predictions = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        self.assertEqual(self.probe_4.n_ps_shifts, 0)

    def test_right_turn_ps_shift(self):
        """Check whether a shift from P to S occurred at right turn."""
        # TFxPS
        self.probe_1.turn_shared = 5
        self.probe_1.predictions = np.array([0, 1, 2, 3, 3, 1, 1, 0, 2, 3, 1])
        self.assertTrue(self.probe_1.right_turn_ps_shift)
        self.probe_1.predictions = np.array([0, 1, 2, 3, 2, 1, 1, 0, 2, 3, 1])
        self.assertTrue(self.probe_1.right_turn_ps_shift)
        self.probe_1.predictions = np.array([0, 1, 2, 3, 0, 1, 1, 0, 2, 3, 1])
        self.assertFalse(self.probe_1.right_turn_ps_shift)
        self.probe_1.predictions = np.array([0, 1, 2, 3, 1, 1, 1, 0, 2, 3, 1])
        self.assertFalse(self.probe_1.right_turn_ps_shift)
        self.probe_1.turn_shared = 0
        self.probe_1.predictions = np.array([0, 1, 2, 3, 1, 1, 1, 0, 2, 3, 1])
        self.assertTrue(self.probe_1.right_turn_ps_shift)
        self.probe_1.predictions = np.array([1, 1, 2, 3, 1, 1, 1, 0, 2, 3, 1])
        self.assertTrue(self.probe_1.right_turn_ps_shift)
        self.probe_1.predictions = np.array([2, 1, 2, 3, 1, 1, 1, 0, 2, 3, 1])
        self.assertFalse(self.probe_1.right_turn_ps_shift)
        self.probe_1.predictions = np.array([3, 1, 2, 3, 1, 1, 1, 0, 2, 3, 1])
        self.assertFalse(self.probe_1.right_turn_ps_shift)
        # TF
        self.probe_2.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        with self.assertRaises(AssertionError):
            _ = self.probe_2.right_turn_ps_shift
        # PS
        self.probe_3.turn_shared = 10
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.assertFalse(self.probe_3.right_turn_ps_shift)
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        self.assertFalse(self.probe_3.right_turn_ps_shift)
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1])
        self.assertFalse(self.probe_3.right_turn_ps_shift)
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0])
        self.assertTrue(self.probe_3.right_turn_ps_shift)
        self.probe_3.turn_shared = 0
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.assertTrue(self.probe_3.right_turn_ps_shift)
        self.probe_3.predictions = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.assertFalse(self.probe_3.right_turn_ps_shift)
        # PxSTSF
        self.probe_4.turn_shared = 2
        self.probe_4.predictions = np.array([0, 1, 2, 2, 1, 0, 2, 0, 1, 0, 1])
        self.assertFalse(self.probe_4.right_turn_ps_shift)
        self.probe_4.predictions = np.array([0, 0, 2, 2, 1, 0, 2, 0, 1, 0, 1])
        self.assertFalse(self.probe_4.right_turn_ps_shift)
        self.probe_4.predictions = np.array([0, 1, 1, 2, 1, 0, 2, 0, 1, 0, 1])
        self.assertFalse(self.probe_4.right_turn_ps_shift)
        self.probe_4.predictions = np.array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0, 1])
        self.assertTrue(self.probe_4.right_turn_ps_shift)
        self.probe_4.predictions = np.array([0, 2, 0, 2, 1, 0, 2, 0, 1, 0, 1])
        self.assertTrue(self.probe_4.right_turn_ps_shift)
        self.probe_4.turn_shared = 0
        self.probe_4.predictions = np.array([1, 2, 0, 2, 1, 0, 2, 0, 1, 0, 1])
        self.assertTrue(self.probe_4.right_turn_ps_shift)
        self.probe_4.predictions = np.array([0, 2, 0, 2, 1, 0, 2, 0, 1, 0, 1])
        self.assertTrue(self.probe_4.right_turn_ps_shift)
        self.probe_4.predictions = np.array([2, 2, 0, 2, 1, 0, 2, 0, 1, 0, 1])
        self.assertFalse(self.probe_4.right_turn_ps_shift)

    def test_only_right_turn_ps_shift(self):
        """Check whether a shift from P to S occurred only at right turn."""
        # TFxPS
        self.probe_1.turn_shared = 5
        self.probe_1.predictions = np.array([3, 3, 2, 3, 3, 1, 1, 0, 0, 1, 1])
        self.assertTrue(self.probe_1.only_right_turn_ps_shift)
        self.probe_1.predictions = np.array([3, 3, 2, 3, 2, 0, 1, 0, 0, 1, 1])
        self.assertTrue(self.probe_1.only_right_turn_ps_shift)
        self.probe_1.predictions = np.array([3, 1, 2, 3, 2, 0, 1, 0, 0, 1, 1])
        self.assertFalse(self.probe_1.only_right_turn_ps_shift)
        self.probe_1.predictions = np.array([3, 3, 2, 3, 3, 1, 1, 0, 0, 1, 2])
        self.assertFalse(self.probe_1.only_right_turn_ps_shift)
        self.probe_1.turn_shared = 0
        self.probe_1.predictions = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        self.assertTrue(self.probe_1.only_right_turn_ps_shift)
        self.probe_1.predictions = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        self.assertTrue(self.probe_1.only_right_turn_ps_shift)
        self.probe_1.predictions = np.array([1, 1, 2, 2, 3, 3, 2, 2, 2, 2, 2])
        self.assertFalse(self.probe_1.only_right_turn_ps_shift)
        self.probe_1.predictions = np.array([2, 3, 3, 2, 2, 2, 3, 3, 2, 2, 3])
        self.assertFalse(self.probe_1.only_right_turn_ps_shift)
        self.probe_1.predictions = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        self.assertFalse(self.probe_1.only_right_turn_ps_shift)
        # TF
        self.probe_2.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        with self.assertRaises(AssertionError):
            _ = self.probe_2.only_right_turn_ps_shift
        # PS
        self.probe_3.turn_shared = 10
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.assertFalse(self.probe_3.only_right_turn_ps_shift)
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        self.assertFalse(self.probe_3.only_right_turn_ps_shift)
        self.probe_3.predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertFalse(self.probe_3.only_right_turn_ps_shift)
        self.probe_3.predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        self.assertTrue(self.probe_3.only_right_turn_ps_shift)
        self.probe_3.turn_shared = 0
        self.probe_3.predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(self.probe_3.only_right_turn_ps_shift)
        self.probe_3.predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertFalse(self.probe_3.only_right_turn_ps_shift)
        self.probe_3.predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertFalse(self.probe_3.only_right_turn_ps_shift)
        # PxSTSF
        self.probe_4.turn_shared = 2
        self.probe_4.predictions = np.array([2, 2, 1, 1, 1, 0, 2, 0, 1, 0, 1])
        self.assertFalse(self.probe_4.only_right_turn_ps_shift)
        self.probe_4.predictions = np.array([1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        self.assertFalse(self.probe_4.only_right_turn_ps_shift)
        self.probe_4.predictions = np.array([2, 2, 1, 1, 1, 0, 0, 0, 1, 0, 1])
        self.assertTrue(self.probe_4.only_right_turn_ps_shift)
        self.probe_4.turn_shared = 0
        self.probe_4.predictions = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
        self.assertTrue(self.probe_4.only_right_turn_ps_shift)
        self.probe_4.predictions = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
        self.assertTrue(self.probe_4.only_right_turn_ps_shift)
        self.probe_4.predictions = np.array([1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 1])
        self.assertFalse(self.probe_4.only_right_turn_ps_shift)
        self.probe_4.predictions = np.array([0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        self.assertFalse(self.probe_4.only_right_turn_ps_shift)
        self.probe_4.predictions = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        self.assertFalse(self.probe_4.only_right_turn_ps_shift)

    def test_is_tf_stable(self):
        """Check whether no shifts occurred in the TF dimension."""
        # TFxPS
        self.probe_1.predictions = np.array([0, 3, 2, 3, 3, 1, 1, 0, 0, 1, 1])
        self.assertFalse(self.probe_1.is_tf_stable)
        self.probe_1.predictions = np.array([0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2])
        self.assertTrue(self.probe_1.is_tf_stable)
        self.probe_1.predictions = np.array([3, 1, 1, 1, 3, 3, 3, 1, 1, 3, 3])
        self.assertTrue(self.probe_1.is_tf_stable)
        # TF
        self.probe_2.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.assertFalse(self.probe_2.is_tf_stable)
        self.probe_2.predictions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(self.probe_2.is_tf_stable)
        self.probe_2.predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertTrue(self.probe_2.is_tf_stable)
        # PS
        self.probe_3.predictions = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
        with self.assertRaises(AssertionError):
            _ = self.probe_3.is_tf_stable
        # PxSTSF
        self.probe_4.predictions = np.array([2, 2, 1, 1, 1, 0, 0, 0, 1, 0, 1])
        self.assertFalse(self.probe_4.is_tf_stable)
        self.probe_4.predictions = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1])
        self.assertFalse(self.probe_4.is_tf_stable)
        self.probe_4.predictions = np.array([2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(self.probe_4.is_tf_stable)
        self.probe_4.predictions = np.array([2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1])
        self.assertTrue(self.probe_4.is_tf_stable)


class TestDialogue(unittest.TestCase):
    """Test methods in Dialogue class."""
    
    def setUp(self):
        self.p_1 = {'proposition': 'there is no dog.', 'a_thinks_true': 1,
                     'turn_shared': 4, 'rule': 'is_it', 'qa_fact': 'negative'}
        self.p_2 = {'proposition': 'there is a dog.', 'a_thinks_true': 0,
                     'turn_shared': 7, 'rule': 'is_it', 'qa_fact': 'positive'}
        self.labels = {0: 'a_thinks_true and shared',
                       1: 'a_thinks_false and shared',
                       2: 'a_thinks_true and private',
                       3: 'a_thinks_false and private'}
        self.probe_1 = Proposition(1, 2, self.p_1, self.labels, 
                                       'a', 'TFxPS')
        self.probe_2 = Proposition(1, 6, self.p_2, self.labels, 
                                       'a', 'TFxPS')
        self.dialogue = Dialogue(123, self.labels)

    def test_add_proposition(self):
        self.assertEqual(self.dialogue.propositions, [])
        self.assertIs(self.dialogue.scoreboard, None)
        self.assertIs(self.dialogue.gold_scoreboard, None)
        self.assertIs(self.dialogue.tf_scoreboard, None)
        self.assertIs(self.dialogue.gold_tf_scoreboard, None)
        self.assertIs(self.dialogue.ps_scoreboard, None)
        self.assertIs(self.dialogue.gold_ps_scoreboard, None)
        
        self.probe_1.gold = np.array([1, 2, 3, 2, 2, 1, 1, 0, 0, 0, 1])
        self.probe_1.predictions = np.array([3, 0, 1, 2, 2, 1, 1, 3, 2, 0, 1])
        self.dialogue.add_proposition(self.probe_1)
        self.assertEqual(self.dialogue.propositions, [self.probe_1])
        self.assertIsNot(self.dialogue.scoreboard, None)
        npt.assert_equal(self.dialogue.gold_scoreboard, self.probe_1.gold)
        npt.assert_equal(self.dialogue.scoreboard, self.probe_1.predictions)

    def test_initialize_scoreboard(self):
        self.assertIs(self.dialogue.scoreboard, None)
        self.assertIs(self.dialogue.gold_scoreboard, None)
        self.assertIs(self.dialogue.tf_scoreboard, None)
        self.assertIs(self.dialogue.gold_tf_scoreboard, None)
        self.assertIs(self.dialogue.ps_scoreboard, None)
        self.assertIs(self.dialogue.gold_ps_scoreboard, None)

        self.dialogue._initialize_scoreboard(self.probe_1) 

        self.assertIsNot(self.dialogue.scoreboard, None)
        self.assertIsNot(self.dialogue.gold_scoreboard, None)
        self.assertIsNot(self.dialogue.tf_scoreboard, None)
        self.assertIsNot(self.dialogue.gold_tf_scoreboard, None)
        self.assertIsNot(self.dialogue.ps_scoreboard, None)
        self.assertIsNot(self.dialogue.gold_ps_scoreboard, None)

    def test_extend_scoreboard(self):
        self.probe_1.gold = np.array([1, 2, 3, 2, 2, 1, 1, 0, 0, 0, 1])
        self.probe_1.predictions = np.array([3, 0, 1, 2, 2, 1, 1, 3, 2, 0, 1])
        self.dialogue.add_proposition(self.probe_1)

        self.probe_2.gold = np.array([1, 2, 3, 2, 0, 1, 1, 0, 0, 1, 2])
        self.probe_2.predictions = np.array([3, 0, 1, 2, 2, 3, 1, 0, 2, 0, 2])
        self.dialogue._extend_scoreboard(self.probe_2)

        scoreboard = np.vstack([self.probe_1.predictions, self.probe_2.predictions])
        gold = np.vstack([self.probe_1.gold, self.probe_2.gold])
        npt.assert_equal(scoreboard, self.dialogue.scoreboard)
        npt.assert_equal(gold, self.dialogue.gold_scoreboard)

    def test_build_rows(self):
        self.probe_1.predictions = np.array([0, 0, 1, 1, 2, 3, 2, 1, 1, 1, 2])
        self.probe_1.gold =        np.array([0, 0, 2, 1, 2, 0, 2, 1, 2, 1, 1])

        tf_p = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1])
        tf_l = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
        ps_p = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0])
        ps_l = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1])
        tf_preds, tf_labels, ps_preds, ps_labels = self.dialogue._build_rows(
            self.probe_1)
        npt.assert_equal(tf_preds, tf_p)
        npt.assert_equal(tf_labels, tf_l)
        npt.assert_equal(ps_preds, ps_p)
        npt.assert_equal(ps_labels, ps_l)
    
    def test_is_empty(self):
        self.assertIsNone(self.dialogue.scoreboard)
        self.dialogue.add_proposition(self.probe_1)
        self.assertIsNot(self.dialogue.scoreboard, None)

    def test_acc(self):
        self.dialogue.scoreboard = np.array([[0, 1, 3],
                                             [0, 1, 2],
                                             [1, 2, 2]])
        self.dialogue.gold_scoreboard = np.array([[0, 1, 3],
                                                  [0, 1, 2],
                                                  [1, 2, 2]])
        self.assertEqual(self.dialogue.acc, 1.0)
        self.dialogue.gold_scoreboard = np.array([[0, 1, 0],
                                                  [1, 1, 2],
                                                  [1, 2, 3]])
        self.assertAlmostEqual(self.dialogue.acc, 0.6666666666666666)

    def test_acc_per_turn(self):
        self.dialogue.scoreboard = np.array([[0, 1, 3],
                                             [0, 1, 2],
                                             [1, 2, 2]])
        self.dialogue.gold_scoreboard = np.array([[0, 1, 3],
                                                  [0, 1, 2],
                                                  [1, 2, 2]])
        accs = np.array([1.0, 1.0, 1.0])
        npt.assert_equal(accs, self.dialogue.acc_per_turn)
        self.dialogue.gold_scoreboard = np.array([[0, 1, 0],
                                                  [1, 1, 2],
                                                  [1, 2, 3]])
        accs = np.array([0.6666666666666666, 1.0, 0.3333333333333333])
        npt.assert_allclose(accs, self.dialogue.acc_per_turn)

    def test_tf_acc(self):
        self.dialogue.tf_scoreboard = np.array([[0, 1, 1],
                                                [0, 1, 0],
                                                [1, 0, 0]])
        self.dialogue.gold_tf_scoreboard = np.array([[0, 1, 1],
                                                     [0, 1, 0],
                                                     [1, 0, 0]])
        self.assertEqual(self.dialogue.tf_acc, 1.0)
        self.dialogue.gold_tf_scoreboard = np.array([[0, 0, 0],
                                                     [0, 1, 0],
                                                     [1, 1, 0]])
        self.assertAlmostEqual(self.dialogue.tf_acc, 0.6666666666666666)

    def test_tf_acc_per_turn(self):
        self.dialogue.tf_scoreboard = np.array([[0, 1, 1],
                                                [0, 1, 0],
                                                [1, 0, 0]])
        self.dialogue.gold_tf_scoreboard = np.array([[0, 1, 1],
                                                     [0, 1, 0],
                                                     [1, 0, 0]])
        accs = np.array([1.0, 1.0, 1.0])
        npt.assert_equal(accs, self.dialogue.tf_acc_per_turn)
        self.dialogue.gold_tf_scoreboard = np.array([[0, 0, 0],
                                                     [0, 1, 1],
                                                     [1, 1, 1]])
        accs = np.array([1.0, 0.3333333333333333, 0.0])
        npt.assert_allclose(accs, self.dialogue.tf_acc_per_turn)

    def test_ps_acc(self):
        self.dialogue.ps_scoreboard = np.array([[0, 1, 1],
                                                [0, 1, 0],
                                                [1, 0, 0]])
        self.dialogue.gold_ps_scoreboard = np.array([[0, 1, 1],
                                                     [0, 1, 0],
                                                     [1, 0, 0]])
        self.assertEqual(self.dialogue.ps_acc, 1.0)
        self.dialogue.gold_ps_scoreboard = np.array([[0, 0, 0],
                                                     [0, 1, 0],
                                                     [1, 1, 0]])
        self.assertAlmostEqual(self.dialogue.ps_acc, 0.6666666666666666)

    def test_ps_acc_per_turn(self):
        self.dialogue.ps_scoreboard = np.array([[0, 1, 1],
                                                [0, 1, 0],
                                                [1, 0, 0]])
        self.dialogue.gold_ps_scoreboard = np.array([[0, 1, 1],
                                                     [0, 1, 0],
                                                     [1, 0, 0]])
        accs = np.array([1.0, 1.0, 1.0])
        npt.assert_equal(accs, self.dialogue.ps_acc_per_turn)
        self.dialogue.gold_ps_scoreboard = np.array([[0, 0, 0],
                                                     [0, 1, 1],
                                                     [1, 1, 1]])
        accs = np.array([1.0, 0.3333333333333333, 0.0])
        npt.assert_allclose(accs, self.dialogue.ps_acc_per_turn)

    def test_tf_stability(self):
        self.probe_1.predictions = np.array([0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0])
        self.dialogue.add_proposition(self.probe_1)
        self.probe_2.predictions = np.array([3, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1])
        self.dialogue.add_proposition(self.probe_2)
        self.assertEqual(self.dialogue.tf_stability, 1.0)

        self.dialogue_2 = Dialogue(123, self.labels)
        self.probe_1.predictions = np.array([0, 0, 1, 2, 2, 2, 2, 2, 2, 0, 0])
        self.dialogue_2.add_proposition(self.probe_1)
        self.dialogue_2.add_proposition(self.probe_2)
        self.assertEqual(self.dialogue_2.tf_stability, 0.5)

        self.dialogue_3 = Dialogue(123, self.labels)
        self.probe_2.predictions = np.array([3, 1, 1, 3, 3, 3, 0, 1, 1, 1, 1])
        self.dialogue_3.add_proposition(self.probe_1)
        self.dialogue_3.add_proposition(self.probe_2)
        self.assertEqual(self.dialogue_3.tf_stability, 0.0)


if __name__ == '__main__':
    unittest.main()
