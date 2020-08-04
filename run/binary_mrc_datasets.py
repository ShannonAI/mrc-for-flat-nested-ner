#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# binary train/dev/test sets for mrc-ner model.


import os
import sys
import argparse
from data_loader.model_config import Config
from data_loader.mrc_data_loader import MRCNERDataLoader
from data_loader.bert_tokenizer import BertTokenizer4Tagger
from data_loader.mrc_data_processor import Conll03Processor, MSRAProcessor, Onto4ZhProcessor, Onto5EngProcessor, GeniaProcessor, ACE2004Processor, ACE2005Processor, ResumeZhProcessor


def collect_arguments():
    parser = argparse.ArgumentParser(description="Arguments for generating binary mrc-ner train/dev/test sets.")

    # required arguments
    parser.add_argument("--data_dir", required=True, default="/data/mrc-ner/zh_onto", type=str, help="")
    parser.add_argument("--max_seq_length", required=True, default=128, type=int, help="")
    parser.add_argument("--data_sign", required=True, type=str, default="zh_onto", help="[data_sign] corespond the a specific dataset.")
    parser.add_argument("--bert_model", required=True, default="/data/pretrained_ckpt/chinese_L-12_H-768_A-12", type=str, help="")

    parser.add_argument("--data_cache", default=True, action='store_false', help=".")
    parser.add_argument("--do_lower_case", default=False, action='store_true', help="lower case of input tokens.")
    parser.add_argument("--allow_impossible", default=True, action='store_false', help="Whether to allow impossible input examples as input.")
    args = parser.parse_args()

    return args


def main():
    arg_configs = collect_arguments()

    if arg_configs.data_sign == "conll03":
        data_processor = Conll03Processor()
    elif arg_configs.data_sign == "zh_msra":
        data_processor = MSRAProcessor()
    elif arg_configs.data_sign == "zh_onto":
        data_processor = Onto4ZhProcessor()
    elif arg_configs.data_sign == "en_onto":
        data_processor = Onto5EngProcessor()
    elif arg_configs.data_sign == "genia":
        data_processor = GeniaProcessor()
    elif arg_configs.data_sign == "ace2004":
        data_processor = ACE2004Processor()
    elif arg_configs.data_sign == "ace2005":
        data_processor = ACE2005Processor()
    elif arg_configs.data_sign == "resume":
        data_processor = ResumeZhProcessor()
    else:
        raise ValueError("Please Notice that your data_sign DO NOT exits !!!!!")


    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer4Tagger.from_pretrained(arg_configs.bert_model, do_lower_case=arg_configs.do_lower_case)

    dataset_loaders = MRCNERDataLoader(arg_configs, data_processor, label_list, tokenizer, mode="transform_binary_files", allow_impossible=arg_configs.allow_impossible)

    train_features = dataset_loaders.convert_examples_to_features(data_sign="train")
    dev_features = dataset_loaders.convert_examples_to_features(data_sign="dev")
    test_features = dataset_loaders.convert_examples_to_features(data_sign="test")



if __name__ == "__main__":
    main()
