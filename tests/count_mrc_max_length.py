#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: count_mrc_max_length.py

import os
import sys

REPO_PATH="/".join(os.path.realpath(__file__).split("/")[:-2])
print(REPO_PATH)
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import os
from datasets.collate_functions import collate_to_max_length
from torch.utils.data import DataLoader
from tokenizers import BertWordPieceTokenizer
from datasets.mrc_ner_dataset import MRCNERDataset



def main():
    # en datasets
    bert_path = "/data/xiaoya/models/bert_cased_large"
    json_path = "/data/xiaoya/datasets/mrc_ner_datasets/en_conll03_truecase_sent/mrc-ner.train"
    is_chinese = False
    # [test]
    # max length is 227
    # min length is 12
    # avg length is 40.45264986967854
    # [dev]
    # max length is 212
    # min length is 12
    # avg length is 43.42584615384615
    # [train]
    # max length is 201
    # min length is 12
    # avg length is 41.733423545331526

    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file)
    dataset = MRCNERDataset(json_path=json_path, tokenizer=tokenizer,
                            is_chinese=is_chinese, max_length=10000)

    dataloader = DataLoader(dataset, batch_size=1,
                            collate_fn=collate_to_max_length)

    length_lst = []
    for batch in dataloader:
        for tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx in zip(*batch):
            tokens = tokens.tolist()
            length_lst.append(len(tokens))
    print(f"max length is {max(length_lst)}")
    print(f"min length is {min(length_lst)}")
    print(f"avg length is {sum(length_lst)/len(length_lst)}")


if __name__ == '__main__':
    main()