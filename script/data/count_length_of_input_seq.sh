#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# count number of length of input_ids.
#


REPO_PATH=/data/xiaoya/work/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


DATA_DIR=$1
CLIP_LEN=$2
BERT_MODEL_DIR=$3


python3 $REPO_PATH/utils/length_statistic.py \
--data_dir $DATA_DIR \
--bert_model $BERT_MODEL_DIR \
--clip_length $CLIP_LEN
# --do_lower_case


