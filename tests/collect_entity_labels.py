#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: collect_entity_labels.py

import os
import sys

REPO_PATH = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(REPO_PATH)
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from datasets.tagger_ner_dataset import load_data_in_conll, get_labels


def main(data_dir, data_sign, datafile_suffix=".word.bmes"):
    label_collection_set = set()
    for data_type in ["train", "dev", "test"]:
        label_lst = []
        data_path = os.path.join(data_dir, f"{data_type}{datafile_suffix}")
        datasets = load_data_in_conll(data_path)
        for data_item in datasets:
            label_lst.extend(data_item[1])

        label_collection_set.update(set(label_lst))

    print("sum the type of labels: ")
    print(len(label_collection_set))
    print(label_collection_set)

    print("%"*10)
    set_labels = get_labels(data_sign)
    print(len(set_labels))


if __name__ == "__main__":
    data_dir = "/data/xiaoya/datasets/ner/ontonotes5"
    data_sign = "en_onto"
    main(data_dir, data_sign)
