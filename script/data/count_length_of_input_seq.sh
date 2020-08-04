#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# count number of length of input_ids.
#


REPO_PATH=/data/xiaoya/work/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


DATA_DIR=/data/xiaoya/work/datasets/mrc_ner/zh_msra
BERT_MODEL=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12
CLIP_LEN=256


python3 $REPO_PATH/utils/length_statistic.py \
--data_dir $DATA_DIR \
--bert_model $BERT_MODEL \
--clip_length $CLIP_LEN
# --do_lower_case


