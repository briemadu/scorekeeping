#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Diverse auxiliary functions used in main.py and dataloader.py.
"""

import json
import os
from pathlib import Path

import pandas as pd

OUTPUT_DIR = 'outputs'

# abbreviations to yield shorter experiment directory names
encoders = {
    'stsb-bert-base': 'sbb',
    'paraphrase-mpnet-base-v2': 'pmbv2',
    'nli-roberta-base-v2': 'nrbv2',
    'stsb-roberta-base-v2': 'srbv2',
}


def make_logging_dir(params, optim=None):
    """Creates log directory and return name in a naming scheme."""
    bot = f'{params.bot}Bot'
    bot_version = params.bot_version.replace('_', '-')
    task = f'task-{params.task}'
    control_task = f'control-{params.control_task}'
    model = params.classifier
    emb = encoders[params.sent_encoder]
    name = "_".join([bot, bot_version, task, control_task, model, emb])
    if optim:
        name = f'optim{optim}_{name}'
    os.mkdir(Path(OUTPUT_DIR, name))
    return name


def counter_to_percentage(counts):
    """Turns counter into dictionary of proportions."""
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def log_table(train_counter, valid_counter, test_counter, name):
    """Log/return dataframe with the proportion of each label on datasets."""

    percs_train = counter_to_percentage(train_counter)
    percs_valid = counter_to_percentage(valid_counter)
    percs_test = counter_to_percentage(test_counter)

    df = pd.DataFrame(percs_train, index=['train']).T
    df['valid'] = [percs_valid[c] for c in df.index]
    df['test'] = [percs_test[c] for c in df.index]

    directory = Path(OUTPUT_DIR)
    path = directory / name / 'label_proportions.csv'
    df.to_csv(path)

    return df


def log_params(params, name):
    """Log an experiment's parameters as a json file."""
    filtered = {k: v for k, v in params.__dict__.items() if k != 'comet_key'}
    directory = Path(OUTPUT_DIR)
    path = directory / name / 'params'
    with open(path, 'w') as f:
        json.dump(filtered, f)


def log_datapoints(datasets, name):
    """Log dict with information on each datapoint for posterior eval."""
    directory = Path(OUTPUT_DIR)
    for split in datasets:
        file = f'identifiers_{split.split}'
        path = directory / name / file
        with open(path, 'w') as f:
            json.dump(split.datapoints, f)


def log_best_epoch(epoch, name):
    """Log best validation epoch in a txt file."""
    directory = Path(OUTPUT_DIR)
    path = directory / name / 'best_valid_epoch.txt'
    with open(path, 'w') as f:
        f.write(f'{epoch}')


def get_predictions_path(name, mode, epoch):
    """Return path to save model predictions."""
    directory = Path(OUTPUT_DIR)
    path = directory / name / f'results_{mode}_{epoch}.csv'
    return path


def get_reps_path(params, split):
    """Return path to directory where dialogue representations are stored."""
    directory = Path(params.path_to_representations)
    path_r = f'{params.bot}Bot_{params.bot_version}_representations_{split}.h5'
    return directory / path_r


def get_probes_path(params, split):
    """Return path to directory where probe sentences are stored."""
    directory = Path(params.path_to_probes)
    path_p = f'propositions_{split}.json'
    return directory / path_p


def get_embs_path(params):
    """Return path to directory where sentence embeddings are stored."""
    directory = Path(params.path_to_embeddings)
    file = f'embeddings_{params.sent_encoder}.p'
    return directory / file


def get_test_lens_path():
    """Return path to file that stores the varying test dialogue lengths."""
    directory = Path('data')
    file = 'visdial_1.0_test_dialogueLens.txt'
    return directory / file
