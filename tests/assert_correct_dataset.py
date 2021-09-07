#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: assert_correct_dataset.py

import os
import sys

REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(REPO_PATH)
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

import json
from datasets.tagger_ner_dataset import load_data_in_conll
from metrics.functional.tagger_span_f1 import get_entity_from_bmes_lst


def count_entity_with_mrc_ner_format(data_path):
    """
    mrc_data_example:
    {
        "context": "The chemiluminescent ( CL ) response of interferon - gamma - treated U937 ( IFN - U937 ) cells to sensitized target cells has been used to detect red cell , platelet and granulocyte antibodies .",
        "end_position": [
        16,
        18
        ],
        "entity_label": "cell_line",
        "impossible": false,
        "qas_id": "532.1",
        "query": "cell line",
        "span_position": [
            "14;16",
            "7;18"
        ],
        "start_position": [
            14,
            7
            ]
        }
    """
    entity_counter = {}
    with open(data_path, encoding="utf-8") as f:
        data_lst = json.load(f)

    for data_item in data_lst:
        tmp_entity_type = data_item["entity_label"]
        if len(data_item["end_position"]) != 0:
            if tmp_entity_type not in entity_counter.keys():
                entity_counter[tmp_entity_type] = len(data_item["end_position"])
            else:
                entity_counter[tmp_entity_type] += len(data_item["end_position"])

    print("UNDER MRC-NER format -> ")
    print(entity_counter)


def count_entity_with_sequence_ner_format(data_path, is_nested=False):
    entity_counter = {}
    if not is_nested:
        data_lst = load_data_in_conll(data_path)
        label_lst = [label_item[1] for label_item in data_lst]
        for label_item in label_lst:
            tmp_entity_lst = get_entity_from_bmes_lst(label_item)
            for tmp_entity in tmp_entity_lst:
                tmp_entity_type = tmp_entity[tmp_entity.index("]")+1:]
                if tmp_entity_type not in entity_counter.keys():
                    entity_counter[tmp_entity_type] = 1
                else:
                    entity_counter[tmp_entity_type] += 1
        print("UNDER SEQ format ->")
        print(entity_counter)
    else:
        # genia, ace04, ace05
        pass


def main(mrc_data_dir, seq_data_dir, seq_data_suffix="char.bmes", is_nested=False):
    for data_type in ["train", "dev", "test"]:
        mrc_data_path = os.path.join(mrc_data_dir, f"mrc-ner.{data_type}")
        seq_data_path = os.path.join(seq_data_dir, f"{data_type}.{seq_data_suffix}")

        print("$"*10)
        print(f"{data_type}")
        print("$"*10)
        count_entity_with_mrc_ner_format(mrc_data_path)
        count_entity_with_sequence_ner_format(seq_data_path, is_nested=is_nested)


if __name__ == "__main__":
    mrc_data_dir = "/data/xiaoya/datasets/mrc_ner/zh_msra"
    seq_data_dir = "/data/xiaoya/datasets/ner/msra"
    seq_data_suffix = "char.bmes"
    is_nested = False
    main(mrc_data_dir, seq_data_dir, seq_data_suffix=seq_data_suffix, is_nested=is_nested)