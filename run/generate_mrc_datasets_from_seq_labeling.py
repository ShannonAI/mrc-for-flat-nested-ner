#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# generate datasets from tagger to mrc-ner


import os
import sys
import argparse

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from data_preprocess.generate_mrc_dataset import generate_query_ner_dataset


def collect_arguments():
    parser = argparse.ArgumentParser(description="Arguments for generating MRC-NER datasets")

    # required arguments
    parser.add_argument("--path_to_source_data_file", required=True, type=str, help="data dirs, seperate with ';'")
    parser.add_argument("--path_to_save_mrc_data_file", required=True, type=str, help="data dirs, seperate with ';'")
    parser.add_argument("--dataset_name", required=True, type=str, help="data dirs, seperate with ';'")
    parser.add_argument("--entity_sign", type=str, default="flat", help="type of entities, [flat/nested]")
    parser.add_argument("--query_sign", type=str, default="default", help="")
    args = parser.parse_args()

    return args


def main():
    argument_configs = collect_arguments()

    generate_query_ner_dataset(argument_configs.path_to_source_data_file,
                               argument_configs.path_to_save_mrc_data_file,
                               entity_sign=argument_configs.entity_sign,
                               dataset_name=argument_configs.dataset_name,
                               query_sign=argument_configs.query_sign)


if __name__ == "__main__":
    main()