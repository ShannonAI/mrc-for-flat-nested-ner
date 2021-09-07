#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: classifier.py

import torch.nn as nn
from torch.nn import functional as F


class SingleLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label):
        super(SingleLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        features_output = self.classifier(input_features)
        return features_output


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate, act_func="gelu", intermediate_hidden_size=None):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = F.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = F.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = F.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


class BERTTaggerClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate, act_func="gelu", intermediate_hidden_size=None):
        super(BERTTaggerClassifier, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = F.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = F.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = F.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
