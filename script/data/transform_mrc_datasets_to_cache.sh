#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# transform mrc-ner datasets to binary files

REPO_PATH=/home/lixiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


# configs for transforming mrc-ner datasets to binary files.
# DATA_DIR -> path to mrc-ner datasets.
# MAX_LEN -> max length for mrc-ner datasets.
# DATA_SIGN -> should take the value of [conll03, zh_msra, zh_onto, en_onto, genia, ace2004, ace2005, resume].
# BERT_MODEL_PATH -> path to pre-trained bert model dir.
DATA_SIGN=$1
MAX_LEN=$2
DATA_DIR=$3
NUM_PROCESSOR=$4
BERT_MODEL_PATH=$5


python3 $REPO_PATH/run/binary_mrc_datasets.py \
--data_dir $DATA_DIR \
--max_seq_length $MAX_LEN \
--data_sign $DATA_SIGN \
--num_data_processor $NUM_PROCESSOR \
--bert_model $BERT_MODEL_PATH