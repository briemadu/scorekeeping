#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Main script to train and evaluate the probing classifier on different tasks.
"""

from pathlib import Path
import time

from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataloader import ProbingDataset
from models import ShallowClassifier, DeeperClassifier, DeepestClassifier
from evaluate import Evaluator, EpochEvaluator
from aux import (make_logging_dir, log_params, log_datapoints, log_table,
                 log_best_epoch)


# _______________________________ SET UP ______________________________________

params = config.args()
name = make_logging_dir(params)
experiment = Experiment(api_key=params.comet_key,
                        project_name=params.comet_project,
                        workspace=params.comet_workspace,
                        disabled=params.ignore_comet
                        )

torch.manual_seed(params.random_seed)
device = torch.device("cpu")
if 'cuda' in params.device and torch.cuda.is_available():
    device = torch.device(params.device)
    torch.cuda.manual_seed_all(params.random_seed)
print(f'Using {device}')

print('Reading corpus...')
trainset = ProbingDataset(params, 'train')
validset = ProbingDataset(params, 'val')
testset = ProbingDataset(params, 'test')
nlabels = len(trainset.label_names)
labels_names = [trainset.label_names[i] for i in range(nlabels)]
print(f'Datapoints: {len(trainset)} (train), {len(validset)} (valid), \
{len(testset)} (test).')

train_loader = DataLoader(trainset, batch_size=params.batch_size, shuffle=True)
valid_loader = DataLoader(validset, batch_size=params.batch_size)
test_loader = DataLoader(testset, batch_size=params.batch_size)

models = {'Shallow': ShallowClassifier, 'Deeper': DeeperClassifier,
          'Deepest': DeepestClassifier}
model = models[params.classifier](params, nlabels=nlabels)
model = model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=params.lr)
evaluator = Evaluator(name, params.early_stopping)
criterion = nn.CrossEntropyLoss()

# Log experiment details
log_params(params, name)
log_datapoints([trainset, validset, testset], name)
proportions_table = log_table(trainset.label_counter, validset.label_counter,
                              testset.label_counter, name)

experiment.log_parameters({k: v for k, v in params.__dict__.items()
                           if k != 'comet_key'})
experiment.log_code('models.py')
experiment.log_code('dataloader.py')
experiment.log_code('evaluate.py')
experiment.log_code('aux.py')
experiment.log_table('data_labels.csv', proportions_table)
experiment.log_other('size_train', len(trainset))
experiment.log_other('size_valid', len(validset))
experiment.log_other('size_test', len(testset))

# _____________________________ ITERATION FUNCTION ___________________________

def step(loader, mode):
    """Perform one complete epoch."""
    print(f'\n {mode.title()}:\n')
    epoch_eval = EpochEvaluator()
    # loop over batches, one epoch
    for item_id, representations, probes, labels in tqdm(loader):

        epoch_eval.update_datapoints(item_id.tolist(), labels.tolist())

        representations = representations.to(device)
        probes = probes.to(device)
        labels = labels.to(device)

        output, predicted = model(representations, probes)
        loss = criterion(output, labels)
        epoch_eval.update_loss(loss.item())
        epoch_eval.update_predictions(predicted.cpu().tolist())

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            if params.clip > 0:
                clip_grad_norm_(model.parameters(), params.clip)
            optimizer.step()

    acc = evaluator.eval(mode, epoch, epoch_eval)

    experiment.log_metric('overall_acc', acc)
    experiment.log_metric('total_loss', epoch_eval.total_loss)
    experiment.log_confusion_matrix(y_true=epoch_eval.labels,
                                    y_predicted=epoch_eval.predictions,
                                    labels=labels_names, epoch=epoch,
                                    file_name=f'cf_{mode}_{epoch}')


# _________________________________ TRAINING __________________________________

start_learn = time.time()
print('Learning started.\n')

# check initial random performance
epoch = 0
with experiment.train(), torch.no_grad():
    model.eval()
    step(train_loader, mode='init_train')
with experiment.validate(), torch.no_grad():
    model.eval()
    step(valid_loader, mode='init_valid')

for epoch in range(1, params.n_epochs + 1):
    print(f'Epoch {epoch}')
    with experiment.train():
        model.train()
        step(train_loader, mode='train')
    with experiment.validate(), torch.no_grad():
        model.eval()
        step(valid_loader, mode='valid')

    better_model_found = evaluator.update_best(epoch)
    if better_model_found:
        torch.save(model.state_dict(), Path(f'models/{name}.pt'))

    if evaluator.make_early_stop(epoch):
        break

experiment.log_epoch_end(epoch)
best_val_acc = evaluator.best_valid_performance
experiment.log_metric('best_val_acc', best_val_acc)
experiment.log_metric('best_valid_epoch', evaluator.best_valid_epoch)
log_best_epoch(evaluator.best_valid_epoch, name)

print('Stopped at epoch {}.\n'.format(epoch+1))
elapsed = time.time() - start_learn
print('Learning took {0[0]:.0f}m {0[1]:.0f}s.\n'.format(divmod(elapsed, 60)))

# _________________________________ TESTING ___________________________________

print('\n Testing...')
model.load_state_dict(torch.load(Path(f'models/{name}.pt')))
with experiment.test(), torch.no_grad():
    epoch = 0
    model.eval()
    step(test_loader, mode='test')

experiment.log_model('best_model', f'models/{name}.pt')
experiment.log_asset_folder(Path(f'outputs/{name}/'))
print('\n Done!')
