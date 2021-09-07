#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_tokenizer.py

from transformers import AutoTokenizer


def tokenize_word(model_path, do_lower_case=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, do_lower_case=do_lower_case)
    context = "EUROPEAN"
    cut_tokens = tokenizer.encode(context, add_special_tokens=False, return_token_type_ids=None)
    print(cut_tokens)
    print(type(cut_tokens))
    print(type(cut_tokens[0]))


if __name__ == "__main__":
    model_path = "/data/xiaoya/models/bert_cased_large"
    tokenize_word(model_path)

