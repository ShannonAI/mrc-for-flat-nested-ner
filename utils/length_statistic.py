#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoy li
# statistic for length of train/dev/test datasets

import os
import sys
import json
import argparse
from collections import OrderedDict

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from data_loader.bert_tokenizer import BertTokenizer4Tagger
from data_loader.bert_tokenizer import whitespace_tokenize


def collect_arguments():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--data_dir", required=True, default="/data/mrc-ner/zh_onto", type=str, help="")
    parser.add_argument("--bert_model_dir", required=True, default="/data/pretrained_ckpt/chinese_L-12_H-768_A-12", type=str, help="")
    parser.add_argument("--clip_length", required=True, default=200, type=int)

    # optional
    parser.add_argument("--do_lower_case", default=False, action='store_true', help="lower case of input tokens.")

    args = parser.parse_args()
    return args


def run_analysis_for_input_length(arg_configs):
    tokenizer = BertTokenizer4Tagger.from_pretrained(arg_configs.bert_model_dir, do_lower_case=arg_configs.do_lower_case)
    print("%=%"*15)
    print("data_dir", "--->", arg_configs.data_dir)
    print("bert_model_dir", "--->", arg_configs.bert_model_dir)
    print("clip_length", "--->", arg_configs.clip_length)

    for data_type in ["train", "dev", "test"]:
        print("==="*15)
        print("*** *** *** " * 5 , data_type, "*** *** *** " * 5)

        input_file_path = os.path.join(arg_configs.data_dir, "mrc-ner.{}".format(data_type))
        with open(input_file_path, "r") as f:
            data_instances = json.load(f)
            # data_instances is a list of dict:
            # the keys of one element in data_instances are:
            # query, context,
            summary_of_input_data = tokenize_input_sequence_to_subtokens(data_instances, tokenizer, arg_configs.clip_length)
            for s_k, s_v in summary_of_input_data.items():
                print(s_k, "---> ", s_v)


def tokenize_input_sequence_to_subtokens(examples, tokenizer, clip_length):
    len_of_queries = []
    len_of_contexts = []
    len_of_inputs = []

    summary_of_inputs = OrderedDict()
    oob_counter = 0

    for example_idx, example_item in enumerate(examples):
        context_subtokens_lst = []
        query_item = example_item["query"]
        context_item = example_item["context"]
        query_subtokens = tokenizer.tokenize(query_item)
        context_whitespace_tokens = whitespace_tokenize(context_item)
        for word_item in context_whitespace_tokens:
            tmp_subword_lst = tokenizer.tokenize(word_item)
            context_subtokens_lst.extend(tmp_subword_lst)
        len_of_queries.append(len(query_subtokens))
        len_of_contexts.append(len(context_subtokens_lst))
        len_of_inputs.append(len(query_subtokens)+len(context_subtokens_lst)+3)
        if len(context_subtokens_lst) >= clip_length:
            oob_counter += 1

    summary_of_inputs["max_query"] = max(len_of_queries)
    summary_of_inputs["max_context"] = max(len_of_contexts)
    summary_of_inputs["max_inputs"] = max(len_of_inputs)

    summary_of_inputs["min_query"] = min(len_of_queries)
    summary_of_inputs["min_context"] = min(len_of_contexts)
    summary_of_inputs["min_inputs"] = min(len_of_inputs)

    summary_of_inputs["avg_query"] = sum(len_of_queries) / len(len_of_queries)
    summary_of_inputs["avg_context"] = sum(len_of_contexts) / len(len_of_contexts)
    summary_of_inputs["avg_inputs"] = sum(len_of_inputs) / len(len_of_inputs)

    summary_of_inputs["num_examples"] = len(len_of_queries)
    summary_of_inputs["oob_examples"] = oob_counter

    return summary_of_inputs


def main():
    arg_configs = collect_arguments()
    run_analysis_for_input_length(arg_configs)


if __name__ == "__main__":
    main()


