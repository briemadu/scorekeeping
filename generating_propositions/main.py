#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Manipulate VisDial QA pairs to turn them into entailment and contradictions.
The resulting datasets are saved in the propositions/original/ directory as
JSON files.
"""

import json
import os
import random
import string
from tqdm import tqdm

from lists import filter_1, filter_2, coref_pronouns
from aux import generate_proposition as gen_prop
from aux import propositions_from_caption


PATH_VISDIAL = './data/visual_dialog/'
DIR = 'propositions/original'
SEED = 2021
# maximum number of tokens in proposition, too long propositions are
# probably wrong
MAX_LEN_PROP = 15

random.seed(SEED)
os.mkdir(DIR)
# ignore 'it' because it appears on many valid propositions (e.g. it is sunny)
coref_pronouns = coref_pronouns.difference(set(['it']))

# _______________________________ LOAD FILES _________________________________

datasets = {}
visdial = {}
with open(PATH_VISDIAL + 'visdial_1.0_train.json', 'r') as data:
    visdial['train'] = json.load(data)
    datasets['train'] = visdial['train']['data']['dialogs']
    print('Train: \t {}'.format(len(datasets['train'])))
with open(PATH_VISDIAL + 'visdial_1.0_val.json', 'r') as data:
    visdial['val'] = json.load(data)
    datasets['val'] = visdial['val']['data']['dialogs']
    print('Valid: \t {}'.format(len(datasets['val'])))
with open(PATH_VISDIAL + 'visdial_1.0_test.json', 'r') as data:
    visdial['test'] = json.load(data)
    datasets['test'] = visdial['test']['data']['dialogs']
    print('Test: \t {}'.format(len(datasets['test'])))

corefs = {}
with open('./data/coref/visdial_v1.0_train_corefs.json', 'r') as data:
    corefs['train'] = json.load(data)
with open('./data/coref/visdial_v1.0_val_corefs.json', 'r') as data:
    corefs['val'] = json.load(data)
with open('./data/coref/visdial_v1.0_test_corefs.json', 'r') as data:
    corefs['test'] = json.load(data)


# __________________________________ AUX  ____________________________________


def get_dialogue(vd_dialog, split):
    """Get caption and turns of a dialogue in a dataset split."""
    caption = vd_dialog['caption']
    turns = []

    if split != 'test':
        for qa in vd_dialog['dialog']:
            q = visdial[split]['data']['questions'][qa['question']]
            a = visdial[split]['data']['answers'][qa['answer']]
            turns.append((q, a))
    else:
        for qa in vd_dialog['dialog']:
            q = visdial['test']['data']['questions'][qa['question']]
            try:
                a = visdial['test']['data']['answers'][qa['answer']]
            # last round has no answer, ignore it
            except KeyError:
                a = ''
            turns.append((q, a))
    return caption, turns


def get_coref(index, split):
    """Retrieve a dialogue with replaced pronouns."""
    dialogue = corefs[split][str(index)]
    if not dialogue:
        return None
    coref = [(dialogue[i]['question'], dialogue[i]['answer'])
             for i in dialogue.keys()]
    return coref


def filter_content(caption, dialogue):
    """Check if dialogue contains words that may be profane or offensive."""
    d = " ".join([" ".join([q, a]) for (q, a) in dialogue])
    d = d.translate(str.maketrans('', '', string.punctuation))
    c = caption.translate(str.maketrans('', '', string.punctuation))
    d_str = set(d.split())
    d_str.update(set(c.split()))
    if d_str & filter_1 or d_str & filter_2:
        return True
    return False


def filter_proposition(p):
    """Ignore propositions that are not adequate."""
    p = p.strip('.')
    # too long
    if len(p.split()) > MAX_LEN_PROP:
        return False
    # contains unsolved pronouns
    if set(p.split()).intersection(coref_pronouns):
        return False
    return True


def clean(p):
    """Fix common small problems."""
    p = p.replace('any thing', 'anything').replace('no thing', 'nothing')
    p = p.replace('a water', 'water')
    return p


def add_prop_to_dic(props_dic, index, entry, turn, p):
    """Add a proposition and its info to dictionary."""

    ((entailment, contradiction), polarity), rule = entry
    if filter_proposition(entailment) and filter_proposition(contradiction):

        e = clean(entailment.lower())
        props_dic['dialogues'][index][p] = {}
        props_dic['dialogues'][index][p]['proposition'] = e
        props_dic['dialogues'][index][p]['a_thinks_true'] = 1
        props_dic['dialogues'][index][p]['turn_shared'] = turn
        props_dic['dialogues'][index][p]['rule'] = rule
        props_dic['dialogues'][index][p]['qa_fact'] = polarity
        p += 1

        c = clean(contradiction.lower())
        props_dic['dialogues'][index][p] = {}
        props_dic['dialogues'][index][p]['proposition'] = c
        props_dic['dialogues'][index][p]['a_thinks_true'] = 0
        props_dic['dialogues'][index][p]['turn_shared'] = turn
        props_dic['dialogues'][index][p]['rule'] = rule
        props_dic['dialogues'][index][p]['qa_fact'] = polarity
        p += 1

    return p


# ______________________________ GENERATION __________________________________

filtered = {'train': [], 'val': [], 'test': []}
for split, data in datasets.items():
    props_dic = {
                'orig_data': 'visdial_v1.0',
                'set': split,
                'dialogues': {x: {} for x in range(len(data))}
                }
    for index, item in tqdm(enumerate(data)):
        p = 0
        caption, dialogue = get_dialogue(item, split)
        if filter_content(caption, dialogue):
            # keep test 7347 because it was used in the eval sample
            if (split, index) != ('test', 7347):
                filtered[split].append(index)
                continue
        coref_dialogue = get_coref(index, split) or dialogue
        props_caption = propositions_from_caption(caption)

        if props_caption:
            for entry in props_caption:
                p = add_prop_to_dic(props_dic, index, (entry, 'caption'), 0, p)

        for n, (q, a) in enumerate(dialogue):
            if split == 'test' and a == '':
                continue
            try:
                entry = gen_prop(
                    q, a, coref_dialogue[n][0], coref_dialogue[n][1])
            except:
                print('Error in dialogue {}, turn {}.'.format(index, n))
                print('\t {}? {}.'.format(q, a))
                continue
            if not entry or not entry[0]:
                # output is None when no rule
                # output is [] when caught by a rule but no prop
                continue

            p = add_prop_to_dic(props_dic, index, entry, n+1, p)

    assert len(props_dic['dialogues']) == len(data)
    with open(DIR + '/propositions_'+split+'.json', 'w') as f:
        json.dump(props_dic, f)
    with open(DIR + '/filtered.json', 'w') as f:
        json.dump(filtered, f)
