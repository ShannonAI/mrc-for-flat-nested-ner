#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_query_ner.py

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from models.classifier import MultiNonLinearClassifier


class BertQueryNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQueryNER, self).__init__(config)
        self.bert = BertModel(config)

        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.mrc_dropout,
                                                       intermediate_hidden_size=config.classifier_intermediate_hidden_size)

        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """

        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits
