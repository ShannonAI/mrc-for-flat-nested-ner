#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: flat_inference.sh
#

REPO_PATH=/data/xiaoya/workspace/mrc-for-flat-nested-ner-github
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=conll03
DATA_DIR=/data/xiaoya/datasets/mrc_ner_datasets/en_conll03_truecase_sent
BERT_DIR=/data/xiaoya/models/uncased_L-12_H-768_A-12
MAX_LEN=180
MODEL_CKPT=/data/xiaoya/outputs/mrc_ner/conll03/large_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen180/epoch=1_v7.ckpt
HPARAMS_FILE=/data/xiaoya/outputs/mrc_ner/conll03/large_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen180/lightning_logs/version_0/hparams.yaml


python3 ${REPO_PATH}/inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--flat_ner \
--dataset_sign ${DATA_SIGN}