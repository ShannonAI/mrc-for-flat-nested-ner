#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh
#

REPO_PATH=/data/xiaoya/workspace/mrc-for-flat-nested-ner-github
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=ace04
DATA_DIR=/data/xiaoya/datasets/mrc-for-flat-nested-ner/ace2004
BERT_DIR=/data/xiaoya/models/uncased_L-12_H-768_A-12
MAX_LEN=100
MODEL_CKPT=/data/xiaoya/outputs/mrc_ner/ace2004/debug_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen100/epoch=0.ckpt
HPARAMS_FILE=/data/xiaoya/outputs/mrc_ner/ace2004/debug_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen100/lightning_logs/version_3/hparams.yaml


python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN}