#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Collect the length of each dialogue on VisDial test set (because it varies).
"""

import json


with open('data/visual_dialog/visdial_1.0_test.json', 'r') as data:
    visdial_test = json.load(data)
    testset = visdial_test['data']['dialogs']

with open('propositions/visdial_1.0_test_dialogueLens.txt', 'w') as file:
    for d, dialogue in enumerate(testset):
        # len is in fact n_complete_turns + 1 because the last turn
        # has no answer. However, I don't subtract 1 here because
        # this extra 1 can be counted as the caption in my model
        file.write(str(d) + '\t' + str(len(dialogue['dialog'])) + '\n')
