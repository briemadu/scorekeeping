#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2018 Modhe, Nirbhay and Prabhu, Viraj and Cogswell, 
# Michael and Kottur, Satwik and Das, Abhishek and Lee, Stefan and Parikh, 
# Devi and Batra, Dhruv
# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: BSD

"""
An adaptation from original train.py and evaluate.py in the repository
https://github.com/vmurahari3/visdial-diversity.

This script turns the retrieving_A_states and retrieving_Q_states Jupyter
notebooks into a single script to extract the dialogue representations of
all three models over all VisDial dialogues and all turns. For details
on each step, check the Jupyter notebooks.

It collects the representations and build a numpy object whose dimensions are
(n_dialogues, n_turns, hidden_dim) and saves it as h5 files.

WARNING: Because of the different dialogue lenghts in the test set and the
batch padding, it also created 'fake' representations for the un-used turns
in the test set. The main experiments scripts take care of not using them
because it only accesses the valid turns (see tests.py). However, any other
use of this script should be aware of that!
"""

VERSIONS = ('RL_DIV', 'SL', 'ICCV_RL')
BOTS = ('A', 'Q')
SPLITS = ('train', 'val', 'test')
BATCH_SIZES = {'train': 41, 'val': 48, 'test': 40}
DIM_HIDDEN = 512 # hidden state dimension, fixed
PATH_TO_VISDIAL = '/project/brie/scripts_scorekeeping/generating_propositions/data/visual_dialog/'

# my imports
import h5py
import json
import numpy as np
from tqdm import tqdm
from itertools import product

# their imports
import gc
import random
from six.moves import range

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import options
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from utils import utilities as utils


# Load VisDial dataset
visdial_data = {}
for split in BATCH_SIZES.keys():
    with open(PATH_TO_VISDIAL + f'visdial_1.0_{split}.json', 'r') as data:
        visdial_data[split] = json.load(data)

def check_captions_match(batch, visdial_data):
    """Check that VisDial order is preserved.

    Args:
        batch : one batch in their preprocessed format.
        visdial_data (dictionary): VisDial splits.

    Returns:
        Bool: True if caption order match the VisDial order.
    """
    # first, check the indexed dataset
    for idx in batch['index']:
        # turn encoded tokens back into words
        dataset_caption = dataset[idx]['cap']
        caption_words = [dataset.ind2word[x] for x in dataset_caption if x !=0]
        caption_words = " ".join(caption_words[1:-1])
        # get the original caption with this index in VisDial
        vd_caption = visdial_data[SPLIT]['data']['dialogs'][idx]['caption']
        if set(vd_caption) == set(caption_words):
            check_1 = True
        # check only the beginning for captions that are too long and were cut
        elif dataset[idx]['cap_len'] == 41 and 'UNK' not in caption_words:
            check_1 = caption_words.startswith(vd_caption[:10]) 
        # UNKs and some differences in `` tokenization
        else:  
            diff = set(caption_words) - set(vd_caption)
            check_1 =  diff in ({"'", 'K', '`', 'U', 'N'}, {'U', 'K', 'N', '`'},
                                {"'", '`'} , {'`'}, {'U', 'K', 'N'}, {"'"}) 
    # somewhat redundant, but check the batch itself    
    for n, encoded_caption in enumerate(batch['cap']):
        caption_words = [dataset.ind2word[x] for x in encoded_caption if x !=0]
        caption_words = " ".join(caption_words[1:-1])
        idx = batch['index'][n]
        vd_caption = visdial_data[SPLIT]['data']['dialogs'][idx]['caption']
        if set(vd_caption) == set(caption_words):
            check_2 = True
        elif batch['cap_len'][n] == 41 and 'UNK' not in caption_words:
            check_2 = caption_words.startswith(vd_caption[:10]) 
        else:  
            diff = set(caption_words) - set(vd_caption)
            check_2 =  diff in ({"'", 'K', '`', 'U', 'N'}, {'U', 'K', 'N', '`'},
                                {"'", '`'} , {'`'}, {'U', 'K', 'N'}, {"'"}) 
        
    return check_1 and check_2


for VERSION, BOT, SPLIT in product(VERSIONS, BOTS, SPLITS):

    params = options.readCommandLine()
    # checkpoints
    params['startFrom'] = f'./checkpoints-release/{VERSION}_ABOT.vd'
    params['qstartFrom'] = f'./checkpoints-release/{VERSION}_QBOT.vd'
    params['useGPU'] = True
    
    # .eval() should ignore dropout, but just in case:
    assert params['dropout'] ==  0
    # using their default options
    assert params['useIm'] == 'late'
    assert params['useHistory'] is True
    assert params['numRounds'] == 10

    # Seed rng for reproducibility
    random.seed(params['randomSeed'])
    torch.manual_seed(params['randomSeed'])
    if params['useGPU']:
        torch.cuda.manual_seed_all(params['randomSeed'])

    # setup dataloader
    dlparams = params.copy()
    dlparams['useHistory'] = True
    dlparams['numRounds'] = 10

    # Setup dataloader
    splits = ['train', 'val', 'test']

    dataset = VisDialDataset(dlparams, splits)

    # Params to transfer from dataset
    transfer = ['vocabSize', 'numOptions', 'numRounds']
    for key in transfer:
        if hasattr(dataset, key):
            params[key] = getattr(dataset, key)

    assert params['numRounds'] == 10

    # Always load checkpoint parameters with continue flag
    params['continue'] = True

    excludeParams = ['batchSize', 'visdomEnv', 'startFrom', 'qstartFrom', 
                     'trainMode', 'evalModeList', 'evalSplit', 'inputImg', 
                     'inputQues', 'inputJson', 'evalTitle', 'beamSize', 
                     'enableVisdom', 'visdomServer', 'visdomServerPort',
                     'savePath','saveName']

    if BOT == 'A':
        aBot = None
        if params['startFrom']:
            assert params['startFrom'] == f'./checkpoints-release/{VERSION}_ABOT.vd'
            aBot, loadedParams, _ = utils.loadModel(params, 'abot', overwrite=True)
            assert aBot.encoder.vocabSize == dataset.vocabSize, "Vocab size mismatch!"
            for key in loadedParams:
                params[key] = loadedParams[key]
            aBot.eval()

    elif BOT == 'Q':
        qBot = None
        if params['qstartFrom']:
            assert params['qstartFrom'] == f'./checkpoints-release/{VERSION}_QBOT.vd'
            qBot, loadedParams, _ = utils.loadModel(params, 'qbot', overwrite=True)
            assert qBot.encoder.vocabSize == params[
                'vocabSize'], "Vocab size mismatch!"
            for key in loadedParams:
                params[key] = loadedParams[key]
            qBot.eval()

    for key in excludeParams:
        params[key] = dlparams[key]

    dataset.split = SPLIT
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZES[SPLIT],
        shuffle=False,
        num_workers=0,
        drop_last=True, # should be False, but we selected exact sizes above
        collate_fn=dataset.collate_fn,
        pin_memory=False) # is False by default

    numIterPerEpoch = dataset.numDataPoints[dataset.split] / BATCH_SIZES[SPLIT]
    print('\n%d iter per epoch.' % numIterPerEpoch) 

    assert params['dropout'] == 0
    assert dlparams['dropout'] == 0
    assert params['continue'] == True
    assert params['useIm'] == 'late'

    def batch_iter(dataloader):
        for idx, batch in enumerate(dataloader):
            yield idx, batch

    dialogue_representations = []

    for batchId, batch in tqdm(batch_iter(dataloader)):

        assert check_captions_match(batch, visdial_data)

        gc.collect()
        # Moving current batch to GPU, if available
        if dataset.useGPU:
            batch = {key: v.cuda() if hasattr(v, 'cuda') \
                else v for key, v in batch.items()}

        image = Variable(batch['img_feat'], requires_grad=False, volatile=True)
        caption = Variable(batch['cap'], requires_grad=False, volatile=True)
        captionLens = Variable(batch['cap_len'], requires_grad=False, volatile=True)
        gtQuestions = Variable(batch['ques'], requires_grad=False, volatile=True)
        gtQuesLens = Variable(batch['ques_len'], requires_grad=False, volatile=True)
        gtAnswers = Variable(batch['ans'], requires_grad=False, volatile=True)
        gtAnsLens = Variable(batch['ans_len'], requires_grad=False, volatile=True)
        
        numRounds = params['numRounds']
        assert numRounds == 10

        batch_representations = []
        
        if BOT == 'A':
            # Setting eval modes for aBot and observing caption and image 
            aBot.eval()
            aBot.reset()
            aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)

        elif BOT == 'Q':
            # Setting eval modes for qBot and observing caption and image 
            qBot.eval()
            qBot.reset()
            qBot.observe(-1, caption=caption, captionLens=captionLens)
            # turn -1 for them is our turn 0, Q sees the caption
            # implicit forward pass:
            _ =  qBot.predictImage()
            
            cur_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]
            batch_representations.append(
                cur_dialog_hidden.data.cpu().unsqueeze(1).numpy())
            
        for round in range(numRounds):

            if BOT == 'A':
                # Answerer Forward Pass
                aBot.observe(
                    round,
                    ques=gtQuestions[:, round],
                    quesLens=gtQuesLens[:, round])
                
                # at this point, for round t, A sees image, caption,
                # QAs until t and question t+1
                # see the other notebook for the explanation why this is the point
                _ = aBot.forward()
                
                # their rounds go from -1 to 9, mine go from 0 to 10
                assert len(aBot.encoder.dialogHiddens) == round + 1
                cur_dialog_hidden = aBot.encoder.dialogHiddens[-1][0]
                batch_representations.append(
                    cur_dialog_hidden.data.cpu().unsqueeze(1).numpy())
                assert len(batch_representations) == round + 1
                if len(batch_representations) > 1:
                    # be sure that it's not copying the same tensor
                    assert not np.allclose(batch_representations[-2], 
                                           batch_representations[-1], atol=0.1)  
                
                # answer will be used on the next round
                aBot.observe(
                    round,
                    ans=gtAnswers[:, round],
                    ansLens=gtAnsLens[:, round])

            elif BOT == 'Q':
                # Questioner Forward Pass
                qBot.observe(
                    round,
                    ques=gtQuestions[:, round],
                    quesLens=gtQuesLens[:, round])
                qBot.observe(
                    round,
                    ans=gtAnswers[:, round],
                    ansLens=gtAnsLens[:, round])
                
                _ = qBot.forward()
            
                # their rounds go from -1 to 9, mine go from 0 to 10
                assert len(qBot.encoder.dialogHiddens) == round + 2
                cur_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]
                batch_representations.append(
                    cur_dialog_hidden.data.cpu().unsqueeze(1).numpy())
                assert len(batch_representations) == round + 2
                if len(batch_representations) > 1:
                    # be sure that it's not copying the same tensor
                    assert not np.allclose(batch_representations[-2], 
                                           batch_representations[-1], atol=0.1)      
        
        if BOT == 'A':
            c1 = torch.ones(gtQuestions[:, 0].shape[0], 1) * 11320
            c2 = torch.ones(gtQuestions[:, 0].shape[0], 1) * 13
            c3 = torch.ones(gtQuestions[:, 0].shape[0], 1) * 11321 
            c4 = torch.zeros(gtQuestions[:, 0].shape[0], gtQuestions[:, 0].shape[1]-3)
            dummyQuestions = Variable(torch.cat([c1, c2, c3, c4], dim=1).long(), 
                                      requires_grad=False).cuda()
            dummyQuesLens = Variable(2*torch.ones(gtQuestions[:, 0].shape[0]).long(), 
                                     requires_grad=False).cuda() 
            aBot.observe(
                    round+1, # pseudo 10th round for them, 11h for us
                    ques=dummyQuestions,
                    quesLens=dummyQuesLens)
            _ = aBot.forward()

            assert len(aBot.encoder.dialogHiddens) == round + 2
            # (batch_size, 1, DIM_HIDDEN)
            cur_dialog_hidden = aBot.encoder.dialogHiddens[-1][0]
            batch_representations.append(
                cur_dialog_hidden.data.cpu().unsqueeze(1).numpy())
            assert len(batch_representations) == numRounds + 1

        torch.cuda.empty_cache()
        
        # (batch_size, numRounds, DIM_HIDDEN)
        batch_stack = np.column_stack(batch_representations)
        assert batch_stack.shape == (BATCH_SIZES[SPLIT], numRounds + 1, DIM_HIDDEN)
        # if any duplicates, size of unique would be smaller
        assert np.unique(batch_stack, axis=0).shape == batch_stack.shape
        assert np.unique(batch_stack, axis=1).shape == batch_stack.shape
        assert np.unique(batch_stack, axis=2).shape == batch_stack.shape
        
        # check that the column stack keeps the right order of turns, 
        # while also checking that the states collected by us round by round 
        # match the final state of the bots internal list of states
        for s in range(11):
            if BOT == 'A':
                assert np.array_equal(batch_stack[:, s], 
                        aBot.encoder.dialogHiddens[s][0].cpu().data.numpy())
            elif BOT == 'Q':
                assert np.array_equal(batch_stack[:, s], 
                        qBot.encoder.dialogHiddens[s][0].cpu().data.numpy())
        
        dialogue_representations.append(batch_stack)
        assert len(dialogue_representations) == (batchId + 1)

    full_stack = np.row_stack(dialogue_representations)
    assert full_stack.shape == (len(visdial_data[SPLIT]['data']['dialogs']), 
                                numRounds + 1, DIM_HIDDEN)
    # if any duplicates, size of unique would be smaller
    assert np.unique(full_stack, axis=0).shape == full_stack.shape
    assert np.unique(full_stack, axis=1).shape == full_stack.shape
    assert np.unique(full_stack, axis=2).shape == full_stack.shape

    # sanity check that row_stack kept order
    assert np.array_equal(full_stack[:BATCH_SIZES[SPLIT]], 
                          dialogue_representations[0])
    assert np.array_equal(full_stack[-BATCH_SIZES[SPLIT]:], 
                          dialogue_representations[-1])

    assert dataset.split == SPLIT.lower()

    file_name = f'{BOT.lower()}Bot_{VERSION}_representations_{dataset.split}.h5'
    path = f'extracted_states/{file_name}'
    with h5py.File(path, 'w') as hfile:
        hfile.create_dataset(
            dataset.split+'_dialogue_representations', 
            dtype='float32', 
            data=full_stack)