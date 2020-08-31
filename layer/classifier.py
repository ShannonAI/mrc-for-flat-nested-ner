#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Description:
# 


import torch
import torch.nn as nn 
import torch.nn.functional as F


class SingleLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label):
        super(SingleLinearClassifier, self).__init__()
        self.num_label = num_label 
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        features_output = self.classifier(input_features)

        return features_output


class SingleNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(SingleNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_label)

    def forward(self, input_features):
        input_features = self.dropout(input_features)
        features_output = self.classifier(input_features)
        features_output = F.gelu(features_output)
        return features_output


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label 
        self.classifier1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.classifier2 = nn.Linear(int(hidden_size/2), num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        input_features = self.dropout(input_features)
        features_output1 = self.classifier1(input_features)
        features_output1 = nn.ReLU()(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2 


class BiaffineClassifier(nn.Module):
    def __init__(self, feature1_size, feature2_size, convert_feature_size, output_feature_size, output_size, ):
        super(BiaffineClassifier, self).__init__()

        self.feature1_size = feature1_size
        self.feature2_size = feature2_size
        self.input_feature1 = nn.Linear(feature1_size, convert_feature_size)
        self.input_feature2 = nn.Linear(feature2_size, convert_feature_size)
        self.affine_feature_size = convert_feature_size
        self.output_feature_size = output_feature_size
        self.linear = nn.Linear(convert_feature_size, convert_feature_size * output_feature_size)
        self.map_to_logits = nn.Linear(output_feature_size, output_size)

    def forward(self, input):
        seq_len = input.size()[1]
        feature1_seq = nn.ReLU()(self.input_feature1(input))
        feature2_seq = nn.ReLU()(self.input_feature2(input))
        batch_size = input.size()[0]
        affine = self.linear(feature1_seq) # batch_size, sequence_len,
        affine = affine.view(batch_size, -1, self.affine_feature_size)
        input2 = torch.transpose(feature2_seq, 1, 2)
        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, seq_len, seq_len, self.output_feature_size)
        logits = self.map_to_logits(biaffine)
        return logits



