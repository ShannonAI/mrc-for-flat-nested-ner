# encoding: utf-8


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
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
