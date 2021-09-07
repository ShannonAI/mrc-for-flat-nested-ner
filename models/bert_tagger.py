#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_tagger.py
#

import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from models.classifier import BERTTaggerClassifier


class BertTagger(BertPreTrainedModel):
    def __init__(self, config):
        super(BertTagger, self).__init__(config)
        self.bert = BertModel(config)

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.classifier_sign == "multi_nonlinear":
            self.classifier = BERTTaggerClassifier(self.hidden_size, self.num_labels,
                                                   config.classifier_dropout,
                                                   act_func=config.classifier_act_func,
                                                   intermediate_hidden_size=config.classifier_intermediate_hidden_size)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,):
        last_bert_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        last_bert_layer = last_bert_layer.view(-1, self.hidden_size)
        last_bert_layer = self.dropout(last_bert_layer)
        logits = self.classifier(last_bert_layer)
        return logits