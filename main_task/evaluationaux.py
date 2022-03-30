#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Auxiliary functions for the evaluation notebook.
"""

from pathlib import Path

import csv
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

from config import Parameters
from dataloader import ProbingDataset
from evaluationtypes import Proposition, Dialogue


TURNS = 11  # number of turns in all train and valid dialogues

def load_params(name, path):
    """Load the parameters of an experiment."""
    path = Path(path, name)
    with open(path / 'params', 'r') as f:
        params = json.load(f)
    params['comet_key'] = ''
    return Parameters(**params)


def get_outputs(name, epoch, split, dataset, path):
    """Load the csv file with predictions for all datapoint."""
    split = 'valid' if split == 'val' else split
    outputs = {}
    path = Path(path, name, f'results_{split}_{epoch}.csv')
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for global_id, prediction, gold_label in reader:
            if global_id == 'global_id':
                continue
            outputs[int(global_id)] = (int(prediction), int(gold_label))
            # compare output labels and dataset labels to help
            # make sure that the loaded dataset is the same
            # as the one built and used during training
            assert dataset.datapoints[int(global_id)][-1] == int(gold_label)
    return outputs


def get_data(bot, task, control_task, path, split='val', embs='pmbv2',
                bot_version='RL-DIV', model='Deeper', epoch=None):
    """Get the predictions and golden labels of a model."""
    name = f'{bot}Bot_{bot_version}_task-{task}_control-{control_task}_{model}_{embs}'
    params = load_params(name, path)
    dataset = ProbingDataset(params, split)
    if epoch is None:
        with open(Path(path, name, 'best_valid_epoch.txt'), 'r') as file:
            epoch = int(file.read())
    outputs = get_outputs(name, epoch, split, dataset, path)
    return dataset, outputs


def get_identifiers(bot, version, task, control_task, path, split, 
                    model='Deeper', embs='pmbv2'):
    """Load datapoint identifiers used in an experiment."""
    name = f'{bot}Bot_{version}_task-{task}_control-{control_task}_{model}_{embs}'
    with open(Path(path, name, f'identifiers_{split}'), 'r') as file:
        identifiers = json.load(file)
    return identifiers


def get_probes(split):
    """Get propositions of a VisDial split."""
    path = Path('data', 'propositions', f'propositions_{split}.json')
    with open(path, 'r') as f:
        probes_data = json.load(f)['dialogues']
    return probes_data


def compare_identifiers(datapoints, identifiers):
    """Check that dataset labels are the same as experiment labels."""
    for key, value in datapoints.items():
        if identifiers[str(key)] != list(value):
            return False
    return True


def build_probes(outputs, dataset, bot, task, split):
    """Build a dictionary contatining all proposition objects."""
    probes_data = get_probes(split)
    all_probes = {}
    for d_id, dialogue in probes_data.items():
        d_id = int(d_id)
        for p_id, prop_object in dialogue.items():
            p_id = int(p_id)
            #  make sure no duplicates were created for some reason
            assert (d_id, p_id) not in all_probes, "Duplicate found!"
            probe = Proposition(d_id, p_id, prop_object, dataset.label_names,
                                bot, task)
            all_probes[(d_id, p_id)] = probe

    for global_id, (prediction, gold_label) in outputs.items():
        d_id, p_id, s_id, turn_shown, label = dataset.datapoints[global_id]
        # checking that order was not mixed up
        assert gold_label == label, "There is a mismatch on the labels!"
        probe = all_probes[(d_id, p_id)]
        probe.populate(gold_label, prediction, turn_shown)
        probe.sent_id = s_id

    # ensure that all positions have been populated
    for probe in all_probes.values():
        assert probe.is_fully_populated, "Missing values!"

    return all_probes


def build_dialogues(probes, dataset, split):
    """Build a dictionary containing all reconstructed dialogue objects."""
    dialogues = {}
    for (d_id, _), proposition in probes.items():
        if d_id not in dialogues:
            dialogues[d_id] = Dialogue(d_id, dataset.labels.items())
        dialogues[d_id].add_proposition(proposition)

    probes_data = get_probes(split)
    for d_id, dialogue in probes_data.items():
        n_props = len(dialogue)
        if n_props > 0:
            d = int(d_id)
            # FIXME: these assertions do not work for the test set with
            # varying lenghts != TURNS
            assert n_props == len(dialogues[d].propositions)
            assert dialogues[d].scoreboard.shape == (n_props, TURNS)
            assert dialogues[d].tf_scoreboard.shape == (n_props, TURNS)
            assert dialogues[d].ps_scoreboard.shape == (n_props, TURNS)

    return dialogues


def get_cm(outputs, labels):
    """Build a confusion matrix dataframe."""
    gold = [elem[1] for elem in outputs.values()]
    predictions = [elem[0] for elem in outputs.values()]
    cm = confusion_matrix(gold, predictions, labels=list(range(len(labels))))
    cm = normalize(cm, axis=1, norm='l1')
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    return df_cm


def get_vd_dialogue(d_id, dataset):
    """Retrieve a VisDial dialogue."""
    caption = dataset['data']['dialogs'][d_id]['caption'] + '.'
    turns = []
    for qa in dataset['data']['dialogs'][d_id]['dialog']:
        q = dataset['data']['questions'][qa['question']] + '? '
        a = dataset['data']['answers'][qa['answer']] + '.'
        turns.append((q + a))
    return caption, turns


def acc_per_turn(results):
    """Get accuracy over dialogue turns."""
    turn_accs = np.zeros(shape=(1, TURNS))
    turn_tf_accs = np.zeros(shape=(1, TURNS))
    turn_ps_accs = np.zeros(shape=(1, TURNS))

    for d in results.values():
        turn_accs = np.vstack([turn_accs, d.acc_per_turn])
        turn_tf_accs = np.vstack([turn_tf_accs, d.tf_acc_per_turn])
        turn_ps_accs = np.vstack([turn_ps_accs, d.ps_acc_per_turn])
    # remove initial zeros    
    turn_accs = turn_accs[1:]
    turn_tf_accs = turn_accs[1:]
    turn_ps_accs = turn_accs[1:]

    t_accs = np.mean(turn_accs, axis=0)
    t_std = np.std(turn_accs, axis=0)
    t_ps_accs = np.mean(turn_ps_accs, axis=0)
    t_ps_std = np.std(turn_ps_accs, axis=0)
    t_tf_accs = np.mean(turn_tf_accs, axis=0)
    t_tf_std = np.std(turn_tf_accs, axis=0)

    return t_accs, t_std, t_tf_accs, t_tf_std, t_ps_accs, t_ps_std


def get_acc(results):
    acc = [p.acc for p in results.values()]
    return np.mean(acc)


def get_mean_std_acc(results):
    accs = [p.acc for p in results.values()]
    return np.mean(accs), np.std(accs)


def get_tf_acc(results):
    tf_accs = [p.tf_acc for p in results.values()]
    return np.mean(tf_accs)


def get_mean_std_tf_acc(results):
    tf_accs = [p.tf_acc for p in results.values()]
    return np.mean(tf_accs), np.std(tf_accs)


def get_ps_acc(results):
    ps_accs = [p.ps_acc for p in results.values()]
    return np.mean(ps_accs)


def get_mean_std_ps_acc(results):
    ps_accs = [p.ps_acc for p in results.values()]
    return np.mean(ps_accs), np.std(ps_accs)


def get_mean_std_labels_per_prop(results):
    n_labels = [p.n_labels for p in results.values()]
    return np.mean(n_labels), np.std(n_labels)


def get_mean_std_shifts_per_prop(results):
    shifts_per_prop = [p.n_shifts for p in results.values()]
    return np.mean(shifts_per_prop), np.std(shifts_per_prop)


def get_shifts_at_right_turn(results):
    right_shifts = [1 if p.right_turn_ps_shift else 0 for p in results.values()]
    return np.mean(right_shifts)


def get_shifts_only_at_right_turn(results):
    only_right_shifts = [1 if p.only_right_turn_ps_shift else 0 
                           for p in results.values()]
    return np.mean(only_right_shifts)


def get_tfstable_proportion(results):
    stable = [1 if p.is_tf_stable else 0 for p in results.values()]
    return np.mean(stable)
