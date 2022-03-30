#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Labels and labels names for each task.
"""

from dataclasses import dataclass


@dataclass
class TFxPS:
    """Main task for aBot: true/false vs. private/shared."""
    labels = {
        (1, 1): 0,  # 'a thinks true and shared'
        (0, 1): 1,  # 'a thinks false and shared'
        (1, 0): 2,  # 'a thinks true and private'
        (0, 0): 3,  # 'a thinks false and private'
        }
    names = {0: 'a_thinks_true and shared',   1: 'a_thinks_false and shared',
             2: 'a_thinks_true and private',  3: 'a_thinks_false and private'}


@dataclass
class TF:
    """Simplified task, only for aBot: true/false."""
    labels = {
        (1, 1): 0,  # 'a thinks true and shared' = True
        (0, 1): 1,  # 'a thinks false and shared' = False
        (1, 0): 0,  # 'a thinks true and private' = True
        (0, 0): 1,  # 'a thinks false and private' = False
        }
    names = {0: 'a_thinks_true',  1: 'a_thinks_false'}


@dataclass
class PS:
    """Simplified task: private/shared."""
    labels = {
        (1, 1): 0,  # 'a thinks true and shared' = Shared
        (0, 1): 0,  # 'a thinks false and shared' = Shared
        (1, 0): 1,  # 'a thinks true and private' = Private
        (0, 0): 1,  # 'a thinks false and private' = Private
        }
    names = {0: 'shared',  1: 'private'}


@dataclass
class PxTSFS:
    """Simplified task: private vs. true+shared/false+shared."""
    labels = {
        (1, 1): 0,  # 'a thinks true and shared'
        (0, 1): 1,  # 'a thinks false and shared'
        (1, 0): 2,  # 'a thinks true and private' = Private
        (0, 0): 2,  # 'a thinks false and private' = Private
        }
    names = {0: 'a_thinks_true and shared',  1: 'a_thinks_false and shared',
             2: 'private'}


def get_task_labels(bot, task):
    """Returns dictionary with labels and names according to the main task."""
    assert task in ('TFxPS', 'TF', 'PS', 'PxTSFS'), 'Invalid task!'
    # main task with all 4 status
    if task == 'TFxPS':
        assert bot == 'a', 'task not possible for qBot'
        task_data = TFxPS()
    # when we just care for the true/false status
    elif task == 'TF':
        assert bot == 'a', 'task not possible for qBot'
        task_data = TF()
    # when we just care for the true/false status
    elif task == 'PS':
        task_data = PS()
    # we just care for the true/false status of shared part
    elif task == 'PxTSFS':
        task_data = PxTSFS()

    return task_data.labels, task_data.names
