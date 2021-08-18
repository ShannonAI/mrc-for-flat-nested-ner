#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: tagger_span_f1.py


from pytorch_lightning.metrics.metric import TensorMetric
from metrics.functional.tagger_span_f1 import tagger_span_f1


class TaggerSpanF1(TensorMetric):
    def __init__(self, reduce_group=None, reduce_op=None):
        super(TaggerSpanF1, self).__init__(name="tagger_span_f1", reduce_group=reduce_group, reduce_op=reduce_op)

    def forward(self, start_preds, end_preds, start_label_mask, end_label_mask, ):
        return tagger_span_f1(start_preds, end_preds, start_label_mask, end_label_mask, )
