#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: tagger_span_f1.py

import torch
import torch.nn.functional as F


def transform_predictions_to_labels(sequence_input_lst, wordpiece_mask, idx2label_map, input_type="logit"):
    """
    shape:
        sequence_input_lst: [batch_size, seq_len, num_labels]
        wordpiece_mask: [batch_size, seq_len, ]
    """
    wordpiece_mask = wordpiece_mask.detach().cpu().numpy().tolist()
    if input_type == "logit":
        label_sequence = torch.argmax(F.softmax(sequence_input_lst, dim=2), dim=2).detach().cpu().numpy().tolist()
    elif input_type == "prob":
        label_sequence = torch.argmax(sequence_input_lst, dim=2).detach().cpu().numpy().tolist()
    elif input_type == "label":
        label_sequence = sequence_input_lst.detach().cpu().numpy().tolist()
    else:
        raise ValueError
    output_label_sequence = []
    for tmp_idx_lst, tmp_label_lst in enumerate(label_sequence):
        tmp_wordpiece_mask = wordpiece_mask[tmp_idx_lst]
        tmp_label_seq = []
        for tmp_idx, tmp_label in enumerate(tmp_label_lst):
            if tmp_wordpiece_mask[tmp_idx] != -100:
                tmp_label_seq.append(idx2label_map[tmp_label])
            else:
                tmp_label_seq.append(-100)
        output_label_sequence.append(tmp_label_seq)
    return output_label_sequence


def compute_tagger_span_f1(sequence_pred_lst, sequence_gold_lst):
    sum_true_positive, sum_false_positive, sum_false_negative = 0, 0, 0

    for seq_pred_item, seq_gold_item in zip(sequence_pred_lst, sequence_gold_lst):
        gold_entity_lst = get_entity_from_bmes_lst(seq_gold_item)
        pred_entity_lst = get_entity_from_bmes_lst(seq_pred_item)

        true_positive_item, false_positive_item, false_negative_item = count_confusion_matrix(pred_entity_lst, gold_entity_lst)
        sum_true_positive += true_positive_item
        sum_false_negative += false_negative_item
        sum_false_positive += false_positive_item

    batch_confusion_matrix = torch.tensor([sum_true_positive, sum_false_positive, sum_false_negative], dtype=torch.long)
    return batch_confusion_matrix


def count_confusion_matrix(pred_entities, gold_entities):
    true_positive, false_positive, false_negative = 0, 0, 0
    for span_item in pred_entities:
        if span_item in gold_entities:
            true_positive += 1
            gold_entities.remove(span_item)
        else:
            false_positive += 1
    # these entities are not predicted.
    for span_item in gold_entities:
        false_negative += 1
    return true_positive, false_positive, false_negative


def get_entity_from_bmes_lst(label_list):
    """reuse the code block from
        https://github.com/jiesutd/NCRFpp/blob/105a53a321eca9c1280037c473967858e01aaa43/utils/metric.py#L73
        Many thanks to Jie Yang.
    """
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        if label_list[i] != -100:
            current_label = label_list[i].upper()
        else:
            continue

        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

