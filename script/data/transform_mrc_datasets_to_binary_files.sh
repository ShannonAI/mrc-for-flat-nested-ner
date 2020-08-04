#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# transform mrc-ner datasets to binary files

REPO_PATH=/data/xiaoya/work/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


# configs for transforming mrc-ner datasets to binary files.
# DATA_DIR -> path to mrc-ner datasets.
# MAX_LEN -> max length for mrc-ner datasets.
# DATA_SIGN -> should take the value of [conll03, zh_msra, zh_onto, en_onto, genia, ace2004, ace2005, resume].
# BERT_MODEL_PATH -> path to pre-trained bert model dir.
DATA_DIR=/data/xiaoya/work/datasets/mrc_ner/zh_onto4
MAX_LEN=256
DATA_SIGN=zh_onto
BERT_MODEL_PATH=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12


python3 $REPO_PATH/run/binary_mrc_datasets.py \
--data_dir $DATA_DIR \
--max_seq_length $MAX_LEN \
--data_sign $DATA_SIGN \
--bert_model $BERT_MODEL_PATH