#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author: xiaoya li
# file: count_length_autotokenizer.py

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)


from transformers import AutoTokenizer
from datasets.tagger_ner_dataset import TaggerNERDataset


class OntoNotesDataConfig:
    def __init__(self):
        self.data_dir = "/data/xiaoya/datasets/ner/zhontonotes4"
        self.model_path = "/data/xiaoya/pretrain_lm/chinese_L-12_H-768_A-12"
        self.do_lower_case = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, tokenize_chinese_chars=True)
        # BertWordPieceTokenizer(os.path.join(self.model_path, "vocab.txt"), lowercase=self.do_lower_case)
        self.max_length = 512
        self.is_chinese = True
        self.threshold = 275
        self.data_sign = "zh_onto"
        self.data_file_suffix = "char.bmes"

class ChineseMSRADataConfig:
    def __init__(self):
        self.data_dir = "/data/xiaoya/datasets/ner/msra"
        self.model_path = "/data/xiaoya/pretrain_lm/chinese_L-12_H-768_A-12"
        self.do_lower_case = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, tokenize_chinese_chars=True)
        self.max_length = 512
        self.is_chinese = True
        self.threshold = 275
        self.data_sign = "zh_msra"
        self.data_file_suffix = "char.bmes"


class EnglishOntoDataConfig:
    def __init__(self):
        self.data_dir = "/data/xiaoya/datasets/ner/ontonotes5"
        self.model_path = "/data/xiaoya/pretrain_lm/cased_L-12_H-768_A-12"
        self.do_lower_case = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.max_length = 512
        self.is_chinese = False
        self.threshold = 256
        self.data_sign = "en_onto"
        self.data_file_suffix = "word.bmes"


class EnglishCoNLLDataConfig:
    def __init__(self):
        self.data_dir = "/data/xiaoya/datasets/ner/conll03_truecase_bmes"
        self.model_path = "/data/xiaoya/pretrain_lm/cased_L-12_H-768_A-12"
        if "uncased" in self.model_path:
            self.do_lower_case = True
        else:
            self.do_lower_case = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, do_lower_case=self.do_lower_case)
        self.max_length = 512
        self.is_chinese = False
        self.threshold = 275
        self.data_sign = "en_conll03"
        self.data_file_suffix = "word.bmes"

class EnglishCoNLL03DocDataConfig:
    def __init__(self):
        self.data_dir = "/data/xiaoya/datasets/mrc_ner/en_conll03_doc"
        self.model_path = "/data/xiaoya/pretrain_lm/cased_L-12_H-768_A-12"
        self.do_lower_case = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.max_length = 512
        self.is_chinese = False
        self.threshold = 384
        self.data_sign = "en_conll03"
        self.data_file_suffix = "word.bmes"

def count_max_length(data_sign):
    if data_sign == "zh_onto":
        data_config = OntoNotesDataConfig()
    elif data_sign == "zh_msra":
        data_config = ChineseMSRADataConfig()
    elif data_sign == "en_onto":
        data_config = EnglishOntoDataConfig()
    elif data_sign == "en_conll03":
        data_config = EnglishCoNLLDataConfig()
    elif data_sign == "en_conll03_doc":
        data_config = EnglishCoNLL03DocDataConfig()
    else:
        raise ValueError
    for prefix in ["test", "train", "dev"]:
        print("=*"*15)
        print(f"INFO -> loading {prefix} data. ")
        data_file_path = os.path.join(data_config.data_dir, f"{prefix}.{data_config.data_file_suffix}")

        dataset = TaggerNERDataset(data_file_path,
                                   data_config.tokenizer,
                                   data_sign,
                                   max_length=data_config.max_length,
                                   is_chinese=data_config.is_chinese,
                                   pad_to_maxlen=False,)
        max_len = 0
        counter = 0
        for idx, data_item in enumerate(dataset):
            tokens = data_item[0]
            num_tokens = tokens.shape[0]
            if num_tokens >= max_len:
                max_len = num_tokens
            if num_tokens > data_config.threshold:
                print(num_tokens)
                counter += 1

        print(f"INFO -> Max LEN for {prefix} set is : {max_len}")
        print(f"INFO -> large than {data_config.threshold} is {counter}")



if __name__ == '__main__':
    data_sign = "en_onto"
    # english ontonotes 5.0
    # test: 172
    # dev: 407
    # train: 306
    count_max_length(data_sign)


