# encoding: utf-8


from pytorch_lightning.metrics.metric import TensorMetric
from .functional.query_span_f1 import query_span_f1


class QuerySpanF1(TensorMetric):
    """
    Query Span F1
    Args:
        flat: is flat-ner
    """
    def __init__(self, reduce_group=None, reduce_op=None, flat=False):
        super(QuerySpanF1, self).__init__(name="query_span_f1",
                                          reduce_group=reduce_group,
                                          reduce_op=reduce_op)
        self.flat = flat

    def forward(self, start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels):
        return query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels,
                             flat=self.flat)
