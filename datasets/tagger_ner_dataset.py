#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: tagger_ner_dataset.py

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


def get_labels(data_sign):
    """gets the list of labels for this data set."""
    if data_sign == "zh_onto":
        return ["O", "S-GPE", "B-GPE", "M-GPE", "E-GPE",
                "S-LOC", "B-LOC", "M-LOC", "E-LOC",
                "S-PER", "B-PER", "M-PER", "E-PER",
                "S-ORG", "B-ORG", "M-ORG", "E-ORG",]
    elif data_sign == "zh_msra":
        return ["O", "S-NS", "B-NS", "M-NS", "E-NS",
                "S-NR", "B-NR", "M-NR", "E-NR",
                "S-NT", "B-NT", "M-NT", "E-NT"]
    elif data_sign == "en_onto":
        return ["O", "S-LAW", "B-LAW", "M-LAW", "E-LAW",
                "S-EVENT", "B-EVENT", "M-EVENT", "E-EVENT",
                "S-CARDINAL", "B-CARDINAL", "M-CARDINAL", "E-CARDINAL",
                "S-FAC", "B-FAC", "M-FAC", "E-FAC",
                "S-TIME", "B-TIME", "M-TIME", "E-TIME",
                "S-DATE", "B-DATE", "M-DATE", "E-DATE",
                "S-ORDINAL", "B-ORDINAL", "M-ORDINAL", "E-ORDINAL",
                "S-ORG", "B-ORG", "M-ORG", "E-ORG",
                "S-QUANTITY", "B-QUANTITY", "M-QUANTITY", "E-QUANTITY",
                "S-PERCENT", "B-PERCENT", "M-PERCENT", "E-PERCENT",
                "S-WORK_OF_ART", "B-WORK_OF_ART", "M-WORK_OF_ART", "E-WORK_OF_ART",
                "S-LOC", "B-LOC", "M-LOC", "E-LOC",
                "S-LANGUAGE", "B-LANGUAGE", "M-LANGUAGE", "E-LANGUAGE",
                "S-NORP", "B-NORP", "M-NORP", "E-NORP",
                "S-MONEY", "B-MONEY", "M-MONEY", "E-MONEY",
                "S-PERSON", "B-PERSON", "M-PERSON", "E-PERSON",
                "S-GPE", "B-GPE", "M-GPE", "E-GPE",
                "S-PRODUCT", "B-PRODUCT", "M-PRODUCT", "E-PRODUCT"]
    elif data_sign == "en_conll03":
        return ["O", "S-ORG", "B-ORG", "M-ORG", "E-ORG",
                "S-PER", "B-PER", "M-PER", "E-PER",
                "S-LOC", "B-LOC", "M-LOC", "E-LOC",
                "S-MISC", "B-MISC", "M-MISC", "E-MISC"]
    return ["0", "1"]


def load_data_in_conll(data_path):
    """
    Desc:
        load data in conll format
    Returns:
        [([word1, word2, word3, word4], [label1, label2, label3, label4]),
        ([word5, word6, word7, wordd8], [label5, label6, label7, label8])]
    """
    dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        datalines = f.readlines()
    sentence, labels = [], []

    for line in datalines:
        line = line.strip()
        if len(line) == 0:
            dataset.append((sentence, labels))
            sentence, labels = [], []
        else:
            word, tag = line.split(" ")
            sentence.append(word)
            labels.append(tag)
    return dataset


class TaggerNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        data_path: path to Conll-style named entity dadta file.
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        is_chinese: is chinese dataset
    Note:
        https://github.com/huggingface/transformers/blob/143738214cb83e471f3a43652617c8881370342c/examples/pytorch/token-classification/run_ner.py#L362
        https://github.com/huggingface/transformers/blob/143738214cb83e471f3a43652617c8881370342c/src/transformers/models/bert/modeling_bert.py#L1739
    """
    def __init__(self, data_path, tokenizer: AutoTokenizer, dataset_signature, max_length: int = 512,
                 is_chinese=False, pad_to_maxlen=False, tagging_schema="BMESO", ):
        self.all_data = load_data_in_conll(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen
        self.pad_idx = 0
        self.cls_idx = 101
        self.sep_idx = 102
        self.label2idx = {label_item: label_idx for label_idx, label_item in enumerate(get_labels(dataset_signature))}

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        data = self.all_data[item]
        token_lst, label_lst = tuple(data)
        wordpiece_token_lst, wordpiece_label_lst = [], []

        for token_item, label_item in zip(token_lst, label_lst):
            tmp_token_lst = self.tokenizer.encode(token_item, add_special_tokens=False, return_token_type_ids=None)
            if len(tmp_token_lst) == 1:
                wordpiece_token_lst.append(tmp_token_lst[0])
                wordpiece_label_lst.append(label_item)
            else:
                len_wordpiece = len(tmp_token_lst)
                wordpiece_token_lst.extend(tmp_token_lst)
                tmp_label_lst = [label_item] + [-100 for idx in range((len_wordpiece - 1))]
                wordpiece_label_lst.extend(tmp_label_lst)

        if len(wordpiece_token_lst) > self.max_length - 2:
            wordpiece_token_lst = wordpiece_token_lst[: self.max_length-2]
            wordpiece_label_lst = wordpiece_label_lst[: self.max_length-2]

        wordpiece_token_lst = [self.cls_idx] + wordpiece_token_lst + [self.sep_idx]
        wordpiece_label_lst = [-100] + wordpiece_label_lst + [-100]
        # token_type_ids: segment token indices to indicate first and second portions of the inputs.
        # - 0 corresponds to a "sentence a" token
        # - 1 corresponds to a "sentence b" token
        token_type_ids = [0] * len(wordpiece_token_lst)
        # attention_mask: mask to avoid performing attention on padding token indices.
        # - 1 for tokens that are not masked.
        # - 0 for tokens that are masked.
        attention_mask = [1] * len(wordpiece_token_lst)
        is_wordpiece_mask = [1 if label_item != -100 else -100 for label_item in wordpiece_label_lst]
        wordpiece_label_idx_lst = [self.label2idx[label_item] if label_item != -100 else -100 for label_item in wordpiece_label_lst]

        return [torch.tensor(wordpiece_token_lst, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(wordpiece_label_idx_lst, dtype=torch.long),
                torch.tensor(is_wordpiece_mask, dtype=torch.long)]


