#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Models used to map a tuple with a probe sentence embedding and a dialogue
state representation into the corresponding task labels (combinations of true,
false, shared, private).

Three probing classifiers, which are NNs with:
- ShallowClassifier: one linear layer
- DeeperClassifier: two linear layers and a sigmoid in between
- DeepestClassifier: three linear layers with sigmoid and ReLU in between

All have a softmax function on top for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowClassifier(nn.Module):
    """Probing classifier, 1 layers and cross entropy loss."""
    def __init__(self, params, nlabels):
        """
        Args:
            params (dataclass): Experiment parameters.
            nlabels (int): Number of classification labels.
        """
        super().__init__()
        torch.manual_seed(params.random_seed)

        input_dimension = params.probe_dim + params.dialogue_rep_dim
        self.decoder = nn.Linear(input_dimension, nlabels)

    def forward(self, representations, probes):
        """Perform a forward pass with the input data.

        The dialogue state representations and the probes' embeddings
        are concatenated and then flow through the layers of the
        neural network. A softmax function on top for prediction
        and the cross entropy function is used to estimate the loss.

        Dialogue state representations are vectors with dimension S.
        Probe embeddings are vectors with dimension P.
        The inputs are batches with N datapoints.

        Args:
            representations (torch.Tensor):
                Batch with dialogue state representations, dim=(N, S).
            probes (torch.Tensor):
                Batch with probes embeddings, dim=(N, P).

        Returns:
            tuple: Output scores and predicted labels.
                (torch.Tensor dim=(batch, nlabels), torch.Tensor dim=N)
        """

        # concatenate dialogue representations and probes' embeddings
        # dim = batch_size, (768+512)
        x = torch.cat((representations, probes), dim=1)
        # dim = batch, nlabels
        x = self.decoder(x)
        predicted = torch.argmax(F.log_softmax(x, dim=1), dim=1)

        return x, predicted


class DeeperClassifier(nn.Module):
    """Probing classifier, 2 layers and cross entropy loss."""
    def __init__(self, params, nlabels):
        """
        Args:
            params (dataclass): Experiment parameters.
            nlabels (int): Number of classification labels.
        """
        super().__init__()
        torch.manual_seed(params.random_seed)

        input_dimension = params.probe_dim + params.dialogue_rep_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, params.hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(params.dropout),
            nn.Linear(params.hidden_dim, nlabels)
        )

    def forward(self, representations, probes):
        """Perform a forward pass with the input data.

        The dialogue state representations and the probes' embeddings
        are concatenated and then flow through the layers of the
        neural network. A softmax function on top for prediction
        and the cross entropy function is used to estimate the loss.

        Dialogue state representations are vectors with dimension S.
        Probe embeddings are vectors with dimension P.
        The inputs are batches with N datapoints.

        Args:
            representations (torch.Tensor):
                Batch with dialogue state representations, dim=(N, S).
            probes (torch.Tensor):
                Batch with probes embeddings, dim=(N, P).

        Returns:
            tuple: Output scores and predicted labels.
                (torch.Tensor dim=(batch, nlabels), torch.Tensor dim=N)
        """

        # dim = batch_size, (768+512)
        x = torch.cat((representations, probes), dim=1)
        # dim = batch, nlabels
        x = self.layers(x)
        predicted = torch.argmax(F.log_softmax(x, dim=1), dim=1)

        return x, predicted


class DeepestClassifier(nn.Module):
    """Probing classifier, 3 layers and cross entropy loss."""
    def __init__(self, params, nlabels):
        """
        Args:
            params (dataclass): Experiment parameters.
            nlabels (int): Number of classification labels.
        """
        super().__init__()
        torch.manual_seed(params.random_seed)

        input_dimension = params.probe_dim + params.dialogue_rep_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, params.hidden_dim),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.hidden_dim, params.hidden_dim_2),
            nn.Sigmoid(),
            nn.Dropout(params.dropout),
            nn.Linear(params.hidden_dim_2, nlabels)
        )

    def forward(self, representations, probes):
        """Perform a forward pass with the input data.

        The dialogue state representations and the probes' embeddings
        are concatenated and then flow through the layers of the
        neural network. A softmax function on top for prediction
        and the cross entropy function is used to estimate the loss.

        Dialogue state representations are vectors with dimension S.
        Probe embeddings are vectors with dimension P.
        The inputs are batches with N datapoints.

        Args:
            representations (torch.Tensor):
                Batch with dialogue state representations, dim=(N, S).
            probes (torch.Tensor):
                Batch with probes embeddings, dim=(N, P).

        Returns:
            tuple: Output scores and predicted labels.
                (torch.Tensor dim=(batch, nlabels), torch.Tensor dim=N)
        """

        # dim = batch_size, (768+512)
        x = torch.cat((representations, probes), dim=1)
        # dim = batch, nlabels
        x = self.layers(x)
        predicted = torch.argmax(F.log_softmax(x, dim=1), dim=1)

        return x, predicted
