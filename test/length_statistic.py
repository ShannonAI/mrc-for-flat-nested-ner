#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoy li
# statistic for length of train/dev/test datasets

import os
import sys
import json


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def run_statist_for_input_length(data_dir):
    for data_type in ["train", "dev", "test"]:
        input_file_path = os.path.join(data_dir, "mrc-ner.{}".format(data_type))
        with open(input_file_path, "r") as f:
            data_instances = json.load(f)




if __name__ == "__main__":
    mrc_data_dir = "/data/xiaoya/work/datasets/mrc_ner/zh_msra"

