#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Extract and save sentence embeddings to avoid doing it on the fly.
"""

import json
import os
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

DIR = 'propositions/embeddings'
PATH = 'propositions/downsampled'

os.mkdir(DIR)

for model in ['paraphrase-mpnet-base-v2',
              'stsb-bert-base',
              'stsb-roberta-base-v2',
              'nli-roberta-base-v2',
              ]:

    sent_encoder = SentenceTransformer(model)#, device='cuda:1')
    emb_dic = {}
    for split in ('train', 'val', 'test'):
        if split in ('val', 'test'):
            path = PATH + '-propositions_' + split + '.json'
        else:
            path = PATH + '-balanced-propositions_' + split + '.json'
        with open(path, 'r') as f:
            prop_data = json.load(f)

        for idx_d, dialogue in tqdm(prop_data['dialogues'].items()):
            for idx_p, prop in dialogue.items():
                sent = prop['proposition']
                if sent not in emb_dic:
                    emb_dic[sent] = sent_encoder.encode(sent)

    with open(DIR + '/embeddings_' + model + '.p', 'wb') as file:
        pickle.dump(emb_dic, file)
