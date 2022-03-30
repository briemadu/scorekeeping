#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT


"""
Auxiliary classes for the evaluation notebook.
"""

from itertools import groupby
from os import stat
import numpy as np

TURNS = 11  # number of turns in all train and valid dialogues

class Proposition:
    """Complete scoreboard for one proposition."""
    def __init__(self, dialogue_id, proposition_id, infos, labels, bot, task,
                 split='valid'):
        # ID
        self.split = split
        self.bot = bot
        self.task = task
        self.dialogue_id = dialogue_id
        self.id = proposition_id
        self.sent_id = None  # will be added later
        self.d_len = TURNS
        self.true_labels = [l for l, name in labels.items() if 'true' in name]
        self.shared_labels = [l for l, name in labels.items() if 'shared' in name]
        # attributes
        self.sentence = infos['proposition']
        self.a_thinks_true = infos['a_thinks_true']
        self.rule = infos['rule']
        self.turn_shared = infos['turn_shared']
        self.polarity = infos['qa_fact'] or 'neutral'
        # data, will be populated later
        self.predictions = np.full(self.d_len, np.inf)
        self.gold = np.full(self.d_len, np.inf)

    def populate(self, gold, prediction, turn):
        """Add a prediction and its gold label."""
        assert self.predictions[turn] == np.inf and self.gold[turn] == np.inf
        self.predictions[turn] = prediction
        self.gold[turn] = gold

    @property
    def is_fully_populated(self):
        """Check that all turns have been populated with predictions and gold."""
        if np.inf in self.predictions or np.inf in self.gold:
            return False
        return True

    @property
    def acc(self):
        """Proportion of correct labels throughout all turns."""
        return np.mean(self.predictions == self.gold)

    @property
    def tf_acc(self):
        """Proportion of correct labels only wrt true/false."""
        assert self.task in ('TFxPS', 'TF'), "No TF dimension in current task."
        tf_preds = np.isin(self.predictions, self.true_labels).astype(int)
        tf_labels = np.isin(self.gold, self.true_labels).astype(int)
        return np.mean(tf_preds == tf_labels)

    @property
    def ps_acc(self):
        """Proportion of correct labels only wrt shared/private."""
        assert self.task != 'TF', "No PS dimension in current task."
        ps_preds = np.isin(self.predictions, self.shared_labels).astype(int)
        ps_labels = np.isin(self.gold, self.shared_labels).astype(int)
        return np.mean(ps_preds == ps_labels)

    @property
    def n_labels(self):
        """How many labels were assigned to this proposition."""
        return len(set(self.predictions))

    @property
    def n_shifts(self):
        """How many class shifts occurred in this proposition."""
        return len([list(j) for i, j in groupby(self.predictions)]) - 1

    @property
    def n_tf_shifts(self):
        """How many shifts on the true/false dimension."""
        assert self.task in ('TFxPS', 'TF'), "No TF dimension in current task."
        tf_preds = np.isin(self.predictions, self.true_labels).astype(int)
        return len([list(j) for i, j in groupby(tf_preds)]) - 1

    @property
    def n_ps_shifts(self):
        """How many shifts on the private/shared dimension"""
        assert self.task != 'TF', "No PS dimension in current task."
        ps_preds = np.isin(self.predictions, self.shared_labels).astype(int)
        return len([list(j) for i, j in groupby(ps_preds)]) - 1

    @property
    def right_turn_ps_shift(self):
        """Was there a shift from private to shared on the right turn?"""
        assert self.task != 'TF', "No PS dimension in current task."
        if self.turn_shared == 0:
            # should start as shared
            return True if self.predictions[0] in self.shared_labels else False
        ps_preds = np.isin(self.predictions, self.shared_labels).astype(int)
        if ps_preds[self.turn_shared] == 1 and ps_preds[self.turn_shared - 1] == 0:
            return True
        return False

    @property
    def only_right_turn_ps_shift(self):
        """Was the p->s shift the only ps shift? I.e., a 100% correct case 
        in terms of ps."""
        assert self.task != 'TF', "No PS dimension in current task."
        # no shifts occured in caption proposition and it starts correctly
        # at shared (regarded as equivalent to there being a right shift 
        # from P->S on the overall case)
        if self.turn_shared == 0:
            if self.right_turn_ps_shift and self.n_ps_shifts == 0:
                return True
            return False
        # the only shift is the right p->s shift
        if self.right_turn_ps_shift and self.n_ps_shifts == 1:
            return True
        return False

    @property
    def is_tf_stable(self):
        """Does true/false status change?"""
        assert self.task != 'PS', "No TF dimension in current task."
        if self.task != 'PxSTSF':
            if self.n_tf_shifts == 0:
                return True
            return False
        else:
            # in 'PxSTSF', only shared labels are either true and false,
            # it's tf stable if it does not mix both
            if len(set(self.predictions).intersection(set(self.shared_labels))) == 1:
                return True
            return False


class Dialogue:
    """Complete scoreboard of one dialogue."""
    def __init__(self, dialogue_id, labels, split='valid'):
        self.split = split
        self.dialogue_id = dialogue_id
        self.labels = labels
        self.propositions = []
        self.scoreboard = None
        self.gold_scoreboard = None
        self.tf_scoreboard = None
        self.gold_tf_scoreboard = None
        self.ps_scoreboard = None
        self.gold_ps_scoreboard = None
        self.len = TURNS

    def add_proposition(self, prop):
        """Add a proposition to the dialogue score matrices."""
        self.propositions.append(prop)
        if self.is_empty:
            self._initialize_scoreboard(prop)
        else:
            self._extend_scoreboard(prop)

    def _initialize_scoreboard(self, prop):
        """Initialize all dialogue scoreboards."""
        tf_preds, tf_labels, ps_preds, ps_labels = self._build_rows(prop)
        self.scoreboard = prop.predictions
        self.gold_scoreboard = prop.gold
        self.tf_scoreboard = tf_preds
        self.gold_tf_scoreboard = tf_labels
        self.ps_scoreboard = ps_preds
        self.gold_ps_scoreboard = ps_labels

    def _extend_scoreboard(self, prop):
        """Add new rows to all scoreboards."""
        tf_preds, tf_labels, ps_preds, ps_labels = self._build_rows(prop)
        self.scoreboard = np.vstack([self.scoreboard, prop.predictions])
        self.gold_scoreboard = np.vstack([self.gold_scoreboard, prop.gold])
        self.tf_scoreboard = np.vstack([self.tf_scoreboard, tf_preds])
        self.gold_tf_scoreboard = np.vstack([self.gold_tf_scoreboard, tf_labels])
        self.ps_scoreboard = np.vstack([self.ps_scoreboard, ps_preds])
        self.gold_ps_scoreboard = np.vstack([self.gold_ps_scoreboard, ps_labels])

    @staticmethod
    def _build_rows(prop):
        """Create rows for TF and PS scoreboards."""
        tf_preds = np.isin(prop.predictions, prop.true_labels).astype(int)
        tf_labels = np.isin(prop.gold, prop.true_labels).astype(int)
        ps_preds = np.isin(prop.predictions, prop.shared_labels).astype(int)
        ps_labels = np.isin(prop.gold, prop.shared_labels).astype(int)
        return tf_preds, tf_labels, ps_preds, ps_labels

    @property
    def is_empty(self):
        """Has the dialogue been initialized?"""
        return self.scoreboard is None

    @property
    def acc(self):
        """Proportion of correct labels over whole dialogue."""
        return np.mean(self.scoreboard == self.gold_scoreboard)

    @property
    def acc_per_turn(self):
        """Proportion of correct labels for each turn."""
        return np.mean(self.scoreboard == self.gold_scoreboard, axis=0)

    @property
    def tf_acc(self):
        """Proportion of correct labels wrt the TF dimension over dialogue."""
        return np.mean(self.tf_scoreboard == self.gold_tf_scoreboard)

    @property
    def tf_acc_per_turn(self):
        """Proportion of correct true/false labels for each turn."""
        return np.mean(self.tf_scoreboard == self.gold_tf_scoreboard, axis=0)

    @property
    def ps_acc(self):
        """Proportion of correct labels wrt the PS dimension over dialogue."""
        return np.mean(self.ps_scoreboard == self.gold_ps_scoreboard)

    @property
    def ps_acc_per_turn(self):
        """Proportion of correct private/shared labels for each turn."""
        return np.mean(self.ps_scoreboard == self.gold_ps_scoreboard, axis=0)

    @property
    def tf_stability(self):
        """Proportion of stable TF predictions in this dialogue."""
        return np.mean([p.is_tf_stable for p in self.propositions])

    @property
    def coherence(self):
        """Do opposite propositions have opposite classes?"""
        pass
