# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: query_span_f1
@time: 2020/9/6 20:05
@desc: 

"""


import torch


def query_span_f1(start_logits, end_logits, match_logits, label_mask, match_labels):
    """
    Compute span f1 according to query-based model output
    Args:
        start_logits: [bsz, seq_len, 2]
        end_logits: [bsz, seq_len, 2]
        match_logits: [bsz, seq_len, seq_len]
        label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    label_mask = label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = torch.argmax(start_logits, dim=2).bool()
    # [bsz, seq_len]
    end_preds = torch.argmax(end_logits, dim=2).bool()
    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_preds = match_label_mask & match_preds
    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()
    return torch.stack([tp, fp, fn])
