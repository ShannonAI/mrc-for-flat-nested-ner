# encoding: utf-8


import torch
from utils.bmes_decode import bmes_decode


def query_span_f1(start_preds, end_preds, match_logits, start_label_mask, end_label_mask, match_labels, flat=False):
    """
    Compute span f1 according to query-based model output
    Args:
        start_preds: [bsz, seq_len]
        end_preds: [bsz, seq_len]
        match_logits: [bsz, seq_len, seq_len]
        start_label_mask: [bsz, seq_len]
        end_label_mask: [bsz, seq_len]
        match_labels: [bsz, seq_len, seq_len]
        flat: if True, decode as flat-ner
    Returns:
        span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [bsz, seq_len, seq_len]
    match_preds = match_logits > 0
    # [bsz, seq_len]
    start_preds = start_preds.bool()
    # [bsz, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds
                   & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                   & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len)
                        & end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()
    return torch.stack([tp, fp, fn])


def extract_flat_spans(start_pred, end_pred, match_pred, label_mask):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    pseudo_tag = "TAG"
    pseudo_input = "a"

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"M-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"S-{pseudo_tag}"

    tags = bmes_decode([(pseudo_input, label) for label in bmes_labels])

    return [(tag.begin, tag.end) for tag in tags]


def remove_overlap(spans):
    """
    remove overlapped spans greedily for flat-ner
    Args:
        spans: list of tuple (start, end), which means [start, end] is a ner-span
    Returns:
        spans without overlap
    """
    output = []
    occupied = set()
    for start, end in spans:
        if any(x for x in range(start, end+1)) in occupied:
            continue
        output.append((start, end))
        for x in range(start, end + 1):
            occupied.add(x)
    return output
