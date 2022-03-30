#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
This script generates json files with the same data as the VisDial dialogues
json files, but with (partial) coreference resolution. It tries to solve
pronouns that are relevant for the propositions, but since the coref model
is not perfect, it may still return pronouns or wrong corefs.

Remaining pronouns will be filtered later when propositions are generated.
Wrong corefs that are too long will be ignored (MAX_LEN). Other wrong corefs
will probably be rare enough on the final propositions.

The pronoun 'her' can be both possessive and oblique. We assume that in most
cases it's a possessive form.
"""

import copy
import itertools
import json
import numpy as np
import os
import string

from allennlp.predictors.predictor import Predictor
from tqdm import tqdm
from lists import (PATH_ALLENNLP, pronouns_to_solve, oblique_pronouns_to_solve,
                   possessive_pronouns_to_solve, filter_1, filter_2)

MAX_LEN = 5
PATH_VISDIAL = './data/visual_dialog/'
predictor = Predictor.from_path(PATH_ALLENNLP)

datasets = {}

with open(PATH_VISDIAL + 'visdial_1.0_train.json', 'r') as data:
    visdial_train = json.load(data)
    datasets['train'] = visdial_train['data']['dialogs']
    print('Train: \t {}'.format(len(datasets['train'])))
with open(PATH_VISDIAL + 'visdial_1.0_val.json', 'r') as data:
    visdial_val = json.load(data)
    datasets['val'] = visdial_val['data']['dialogs']
    print('Valid: \t {}'.format(len(datasets['val'])))
with open(PATH_VISDIAL + 'visdial_1.0_test.json', 'r') as data:
    visdial_test = json.load(data)
    datasets['test'] = visdial_test['data']['dialogs']
    print('Test: \t {}'.format(len(datasets['test'])))

solved_corefs = {
    'train': {x: 0 for x in range(len(datasets['train']))},
    'val': {x: 0 for x in range(len(datasets['val']))},
    'test': {x: 0 for x in range(len(datasets['test']))}
    }
fails = {'train': [], 'val': [], 'test': []}
too_long = {'train': 0, 'val': 0, 'test': 0}

# __________________________________ AUX _____________________________________


def cap_dialog_to_string(caption, turns):
    """Merge a caption and a dialogue into a string."""
    s = caption
    ts = " ".join([q + "? " + a + "." for (q, a) in turns])
    return s + ". " + ts


def get_dialogue(vd_dialog, split):
    """
    Getting info from a Visdial dialogue
    returns (caption, dialogue) where dialogue is a list of (q, a) tuples
    """
    caption = vd_dialog['caption']
    turns = []
    if split == 'train':
        for qa in vd_dialog['dialog']:
            q = visdial_train['data']['questions'][qa['question']]
            a = visdial_train['data']['answers'][qa['answer']]
            turns.append((q, a))
    elif split == 'val':
        for qa in vd_dialog['dialog']:
            q = visdial_val['data']['questions'][qa['question']]
            a = visdial_val['data']['answers'][qa['answer']]
            turns.append((q, a))
    elif split == 'test':
        for qa in vd_dialog['dialog']:
            q = visdial_test['data']['questions'][qa['question']]
            try:
                a = visdial_test['data']['answers'][qa['answer']]
            # last round in test has no answer, ignore it and don't include
            except KeyError:
                continue
            turns.append((q, a))
    return caption, turns

# _____________________________ COREFERENCE __________________________________


def adjust_corefqa(orig_caption, orig_dialogue, caption, qas):
    """
    Small adjustments to undo changes done by the coref system.
    """
    solved_caption = orig_caption.replace(' \'', '\'').replace(' - ', '-')
    solved_caption = solved_caption.replace(' ,', ',').replace(' n\'t', 'n\'t')
    solved_caption = solved_caption.replace('  ', ' ').replace(' n’t', 'n’t')
    # an exception occurs in the few cases where a dot is used apart from
    # a sentence delimiter or inside a number.
    # in such cases, we'll use the original dialogue.
    n_turns = len(qas)
    try:
        solved_dialogue = [(qas[i], qas[i+1]) for i in range(0, n_turns, 2)]
    except IndexError:
        return None, None
    # coref solver adds some extra spaces... solve them manually
    for n, (q, a) in enumerate(solved_dialogue):
        adj_q = q.replace(' \'', '\'').replace(' ,', ',')
        adj_q = adj_q.replace(' n\'t', 'n\'t').replace('  ', ' ')
        adj_q = adj_q.replace(' n’t', 'n’t').replace(' ’s', '’s')
        adj_q = adj_q.replace(' !', '!').replace('  ’re', ' ’re').replace(' :', ':')
        adj_a = a.replace(' \'', '\'').replace(' ,', ',')
        adj_a = adj_a.replace(' n\'t', 'n\'t').replace('  ', ' ')
        adj_a = adj_a.replace(' n’t', 'n’t').replace(' ’s', '’s')
        adj_a = adj_a.replace(' !', '!').replace('  ’re', ' ’re').replace(' :', ':')

        if ' - ' in q and ' - ' not in orig_dialogue[n][0]:
            adj_q = adj_q.replace(' - ', '-')
        if ' - ' in a and ' - ' not in orig_dialogue[n][1]:
            adj_a = adj_a.replace(' - ', '-')
        if 'can not' in q and 'can not' not in orig_dialogue[n][0]:
            adj_q = adj_q.replace('can not', 'cannot')
        if 'can not' in a and 'can not' not in orig_dialogue[n][1]:
            adj_a = adj_a.replace('can not', 'cannot')
        solved_dialogue[n] = (adj_q, adj_a)
    return solved_caption, solved_dialogue


def coreftext_to_caption_and_dialogue(new_text, orig_caption, orig_dialogue):
    """
    Output of coref system is a string. Put it back into VisDial format with
    (q,a) pairs.
    """
    # sometimes the coref output it does not split '.',
    # so do it manually to minimize cases of
    # failing because of wrong len
    text = []
    for token in new_text:
        if len(token) > 1 and token[-1] == '.':
            text += [token[:-1], '.']
        else:
            text.append(token)
    # retrieve turns back using .
    # for a few cases it will not work correctly, cases will be caught on
    # adjust corefqa and then original dialogue will be used
    turns = [list(turn) for dot, turn in itertools.groupby(
                                                text, key='.'.__ne__) if dot]
    caption = " ".join(turns[0])
    qas = [" ".join(list(qa)) for turn in turns[1:]
           for dot, qa in itertools.groupby(turn, key='?'.__ne__) if dot]

    adjusted_caption, adjusted_dialogue = adjust_corefqa(
                                    orig_caption, orig_dialogue, caption, qas
                                    )
    return adjusted_caption, adjusted_dialogue


def replace_pronouns(caption, dialogue, split, idx):
    """Solves corefs in a dialogue and return object of same type."""
    # create a single text string
    text = cap_dialog_to_string(caption, dialogue)
    # get coref clusters
    corefs = predictor.predict(text)
    new_text = copy.copy(corefs['document'])
    for cluster in corefs['clusters']:
        # assume that entity is the first mention in the cluster
        [begin_entity, end_entity] = cluster[0]
        # get entiy mention
        entity = " ".join(corefs['document'][begin_entity: end_entity+1])
        entity = entity.replace(' .', '')
        # if the reference is too long, it will probably be a large portion
        # of the caption and cause trouble in the text
        # like 'a man on a white bench weating a jacket's jacket is black'
        # pick an arbitrary limit
        if len(entity.split()) > MAX_LEN:
            too_long[split] += 1
            continue
        # subsequent mentions of an entity shall use definite pronoun
        if entity.startswith('a '):
            entity = entity[2:]
        if entity.startswith('an '):
            entity = entity[3:]
        entity = entity.replace(' ?', '')
        if not entity.startswith(('the', 'this')):
            entity = 'the ' + entity
        # replace all occurrences by entity mention
        for [begin, end] in cluster[1:]:
            pronouns_list = pronouns_to_solve.union(oblique_pronouns_to_solve)
            if begin == end and corefs['document'][begin] in pronouns_list:
                new_text[begin] = entity
                solved_corefs[split][idx] += 1  # bookkeeping
            if begin == end and corefs['document'][begin] in possessive_pronouns_to_solve:
                new_text[begin] = entity + '\'s'
                solved_corefs[split][idx] += 1  # bookkeeping
                # checking how many her appears, possessive or oblique?
                # if corefs['document'][begin] == 'her':
                #   print(corefs['document'])
    # put text back into original object structure
    solved_caption, solved_text = coreftext_to_caption_and_dialogue(
                                                    new_text, caption, dialogue
                                                    )
    return solved_caption, solved_text

def filter_content(caption, dialogue):
    """Check if dialogue contains words that may be innapropriate/offensive."""
    d = " ".join([" ".join([q, a]) for (q, a) in dialogue])
    d = d.translate(str.maketrans('', '', string.punctuation))
    c = caption.translate(str.maketrans('', '', string.punctuation))
    d_str = set(d.split())
    d_str.update(set(c.split()))
    if d_str & filter_1 or d_str & filter_2:
        return True
    return False

# _____________________________ GENERATE _____________________________________


if __name__ == '__main__':
    
    os.mkdir('data/coref')

    for split, dialogues in datasets.items():
        print('\n', split.upper(), '\n')
        corefs = {}
        for idx, item in tqdm(enumerate(dialogues)):
            corefs[idx] = {}
            caption, dialogue = get_dialogue(item, split)
            # ignore dialogues that will have no proposition
            if filter_content(caption, dialogue):
                continue
            # solve corefs
            _, coref_dialogue = replace_pronouns(caption, dialogue, split, idx)
            if coref_dialogue:
                aux_dic = {i: {'question': q, 'answer': a}
                           for i, (q, a) in enumerate(coref_dialogue)}
                if len(aux_dic) == len(dialogue):
                    corefs[idx] = aux_dic
                else:
                    # for some reason dialogue could not be correctly
                    # reconstructed, ignore
                    print('bug in dialogue', idx, 'wrong len')
                    fails[split].append(idx)
            else:
                # if coref solver failed (because of . delimiter), empty dic
                # ignore dialogue
                print('ignoring dialogue', idx, 'delimiter')
                fails[split].append(idx)

        assert len(corefs) == len(dialogues)
        with open('./data/coref/visdial_v1.0_'+split+'_corefs.json', 'w') as f:
            json.dump(corefs, f)

        avg = np.mean(list(solved_corefs[split].values()))
        print(f'Average solved corefs per dialogue: {avg}.')
        n_failed = len(fails[split])
        print(f'Dialogues in which coref failed: {n_failed}.')

    # logs
    with open('./data/coref/stats.json', 'w') as f:
        json.dump(solved_corefs, f)
    with open('./data/coref/fails.json', 'w') as f:
        json.dump(fails, f)
    with open('./data/coref/too_long.json', 'w') as f:
        json.dump(too_long, f)
