#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: model_config.py

from transformers import BertConfig


class BertQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
        self.classifier_intermediate_hidden_size = kwargs.get("classifier_intermediate_hidden_size", 1024)
        self.classifier_act_func = kwargs.get("classifier_act_func", "gelu")

class BertTaggerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertTaggerConfig, self).__init__(**kwargs)
        self.num_labels = kwargs.get("num_labels", 6)
        self.classifier_dropout = kwargs.get("classifier_dropout", 0.1)
        self.classifier_sign = kwargs.get("classifier_sign", "multi_nonlinear")
        self.classifier_act_func = kwargs.get("classifier_act_func", "gelu")
        self.classifier_intermediate_hidden_size = kwargs.get("classifier_intermediate_hidden_size", 1024)