#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Configurations of the experiment that can be passed as command line args.

# Using format idea from:
# https://github.com/batra-mlp-lab/visdial-rl/blob/master/options.py
"""

import argparse
from dataclasses import dataclass


@dataclass
class Parameters:
    """Store all experiment details."""
    path_to_probes: str
    path_to_representations: str
    path_to_embeddings: str

    ignore_comet: bool
    comet_key: str
    comet_project: str
    comet_workspace: str

    random_seed: int
    device: str
    sent_encoder: str
    bot: str
    bot_version: str
    control_task: str
    task: str

    classifier: str
    batch_size: int
    probe_dim: int
    dialogue_rep_dim: int
    hidden_dim: int
    hidden_dim_2: int
    n_epochs: int
    early_stopping: int
    lr: float
    dropout: float
    clip: float


def args():
    """Parse arguments given by user and return a dictionary."""
    parser = argparse.ArgumentParser(
        description='Training and evaluating probing classifier for \
                     Visdial dialogue state representations.')

    # _________________________________ PATHS _________________________________
    parser.add_argument('-path_to_probes', default='data/propositions',
                        type=str,
                        help='Path to folder with probes json files.')
    parser.add_argument('-path_to_representations',
                        default='data/representations',
                        type=str,
                        help='Path to folder with HDF5 files \
                                with the dialogue representations')
    parser.add_argument('-path_to_embeddings', default='data/embeddings',
                        type=str,
                        help='Path to folder with probe embeddings.')

    # _________________________________ COMET _________________________________
    parser.add_argument('-ignore_comet', action='store_true',
                        help='Do not log details to Comet_ml.')
    parser.add_argument('-comet_key', default='',
                        type=str, help='Comet.ml personal key.')
    parser.add_argument('-comet_project', default='scorekeeping-experiments',
                        type=str, help='Comet.ml project name.')
    parser.add_argument('-comet_workspace', default='',
                        type=str, help='Comet.ml workspace name.')

    # ______________________________ SETTING __________________________________
    parser.add_argument('-random_seed', default=54321, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('-device', default='cuda:0', type=str,
                        choices=['cpu', 'cuda:0', 'cuda:1'],
                        help='Which device to use.')
    parser.add_argument('-sent_encoder', default='paraphrase-mpnet-base-v2',
                        type=str, choices=[
                            'stsb-bert-base', 'stsb-roberta-base-v2',
                            'nli-roberta-base-v2', 'paraphrase-mpnet-base-v2',
                            ],
                        help='Which sentence encoder model to used for\
                            probe embeddings.')
    parser.add_argument('-bot', default='a', type=str,
                        choices=['a', 'q', 'sc'],
                        help='a for answerer bot, q for questioner bot.')
    parser.add_argument('-bot_version', default='RL_DIV', type=str,
                        choices=['RL_DIV', 'ICCV_RL', 'SL'],
                        help='Which version of the pretrained bots to use.')
    parser.add_argument('-task', default='TFxPS', type=str,
                        choices=['TFxPS', 'TF', 'PS', 'PxTSFS'],
                        help='Which task to train classifier on:\
                            TFxPS for main task with 4 labels,\
                            TF for true/false only,\
                            PS for private/share only.\
                            PxTSFS for private, true+shared, false+shared.')
    parser.add_argument('-control_task', default='none', type=str,
                        choices=['none', 'rand-reps', 'null-reps'],
                        help='Which control task: \
                            none: real task,\
                            rand-reps: random vectors as dialogue states,\
                            null-reps: null vectors as dialogue states.')

    # __________________________ TRAINING PARAMS ______________________________
    parser.add_argument('-classifier', default='Deeper',
                        choices=['Shallow', 'Deeper', 'Deepest'],
                        type=str, help='Which classifier architecture.')
    parser.add_argument('-batch_size', default=512, type=int,
                        help='Batch size.')
    parser.add_argument('-hidden_dim', default=1024, type=int,
                        help='Classifier hidden layer dimension.')
    parser.add_argument('-probe_dim', default=768, type=int,
                        help='Dimensions of the sentence embeddings.')
    parser.add_argument('-dialogue_rep_dim', default=512, type=int,
                        help='Dimensions of the dialogue representations.')
    parser.add_argument('-hidden_dim_2', default=64, type=int,
                        help='Classifier second hidden layer dimension,\
                            only used on DeepestClassifier.')
    parser.add_argument('-n_epochs', default=30, type=int,
                        help='Number of epochs.')
    parser.add_argument('-early_stopping', default=30, type=int,
                        help='Max iterations for early stopping.')
    parser.add_argument('-lr', default=0.001, type=float,
                        help='Learning rate.')
    parser.add_argument('-dropout', default=0.1, type=float,
                        help='Droupout.')
    parser.add_argument('-clip', default=1, type=float,
                        help='Clipping size, use 0 for no clipping.')

    parameters = Parameters(**vars(parser.parse_args()))

    return parameters
