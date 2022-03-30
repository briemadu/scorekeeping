#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
An object to load datasets and create the datapoints for each set.
It loads probes from JSON files and representations from h5 files
and then constructs datasets for the probing classifier task.
"""

import json
import pickle
from collections import Counter, defaultdict
import h5py
import numpy as np

from torch.utils.data import Dataset
from aux import (get_reps_path, get_probes_path, get_test_lens_path,
                 get_embs_path)
from tasks import get_task_labels

# fixed number of turns in all train/valid dialogues
VISDIAL_LEN = 11


class ProbingDataset(Dataset):
    """Build a dataset split."""
    def __init__(self, params, split):
        """
        Args:
            params (dataclass): All parameters of the experiment.
            split (str): Train, valid or test.
        """
        self.params = params
        self.split = split
        self.labels, self.label_names = self._define_labels()
        self.representations = self._load_representations()
        self.label_counter = Counter()
        self.datapoints = {}
        self._create_datapoints()

    def __len__(self):
        """Return number of datapoints."""
        return len(self.datapoints)

    def __getitem__(self, index):
        """Retrieve a datapoint given its index."""
        dialogue_id, _, sent_id, turn, label = self.datapoints[index]
        sent_embedding = self.id2sent[sent_id]
        representation = self.representations[dialogue_id, turn]
        return (index, representation, sent_embedding, label)

    def _define_labels(self):
        """Get labels and their names according to main task."""
        return get_task_labels(self.params.bot, self.params.task)

    def _load_lens(self):
        """Return dict of dialogue lens, which is constant if not test set."""
        if self.split != 'test':
            return defaultdict(lambda: VISDIAL_LEN)
        path = get_test_lens_path()
        with open(path, 'r') as f:
            lens = {}
            for line in f.readlines():
                idx, length = line.strip('\n').split('\t')
                lens[idx] = int(length)
        return lens

    def _load_representations(self):
        """Load dialogue representations."""
        path = get_reps_path(self.params, self.split)
        name = f'{self.split}_dialogue_representations'
        representations = np.array(h5py.File(path, 'r').get(name))

        # Define control task
        if self.params.control_task == 'rand-reps' and self.split == 'train':
            # replace representations by random vectors
            np.random.seed(self.params.random_seed)
            r_mean = np.mean(representations)
            r_std = np.std(representations)
            representations = np.random.normal(loc=r_mean, scale=r_std,
                size = representations.shape).astype(np.float32)
        if self.params.control_task == 'null-reps' and self.split == 'train':
            # use null vectors to completely ignore dialogue states
            representations = np.zeros(
                representations.shape).astype(np.float32)
        return representations

    def _load_probes(self):
        """Load propositions."""
        path = get_probes_path(self.params, self.split)
        with open(path, 'r') as f:
            probes = json.load(f)
        return probes

    def _load_embeddings(self):
        """Load sentence embeddings."""
        path = get_embs_path(self.params)
        with open(path, 'rb') as file:
            embeddings = pickle.load(file)
        return embeddings

    def _create_datapoints(self):
        """Create dictionary with datapoints.

        Loop over all dialogues, over all probes and create the datasets
        in which all dialogue states and all its probe embeddings are paired
        and assigned the corresponding label.

        Datapoints are stored with unique indexes in the items dictionary
        that is later accessed by the __getitem__ method.
        """
        data = self._load_probes()
        lens = self._load_lens()
        sentences = {}
        for idx_d, dialogue in data['dialogues'].items():
            n_turns = lens[idx_d]
            for idx_p, prop in dialogue.items():
                a_thinks_true = prop['a_thinks_true']
                turn = prop['turn_shared']
                sent = prop['proposition']
                if sent not in sentences:
                    # unique index per probe type
                    sentences[sent] = len(sentences) + 1
                idxs = (idx_d, idx_p, sentences[sent])
                labels = self._create_labels(a_thinks_true, turn, n_turns)
                # each probe is paired with all dialogue turns
                new_items = self._create_new_items(n_turns, idxs, labels)
                # key=global id, value=datapoint information
                self.datapoints.update({x[0]: x[1:] for x in new_items})
                self.label_counter.update([self.label_names[c] for c in labels])

        sent_embeddings = self._load_embeddings()
        self.id2sent = {idx: sent_embeddings[sent]
                        for sent, idx in sentences.items()}

    def _create_labels(self, a_thinks_true, turn, n_turns):
        """Return list of gold labels for a full dialogue.

        Labels can have two dimensions:
            - true/false: is fixed for each probe, on all turns
            - private/shared: a probe is private until it's mentioned
                in the dialogue at the given turn

        Args:
            a_thinks_true (int): 0 if probe is false, 1 if it is true for A.
            turn (int): Turn at which a probe becomes shared.
            n_turns (int): How many turns in current dialogue.

        Returns:
            list: A list of gold labels for all turns.
        """
        labels = [self.labels[(a_thinks_true, 0)] if x < turn
                  else self.labels[(a_thinks_true, 1)]
                  for x in range(n_turns)]
        return labels

    def _create_new_items(self, n_turns, idxs, labels):
        """Return a list with new datapoints."""
        idx_d, idx_p, idx_sent = idxs
        n_items = len(self.datapoints)
        # global id, unique identifier for each element
        ids = list(range(n_items, n_items + n_turns))
        # original dialogue id
        d_ids = [int(idx_d)] * n_turns
        # original proposition/probe id
        p_ids = [int(idx_p)] * n_turns
        # sentence id in dataset class
        s_ids = [idx_sent] * n_turns
        # dialogue turns
        turns = list(range(n_turns))
        return zip(ids, d_ids, p_ids, s_ids, turns, labels)
