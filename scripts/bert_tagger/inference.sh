#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: inference.sh


REPO_PATH=/home/lixiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=en_onto
DATA_DIR=/data/lixiaoya/datasets/bmes_ner/en_ontonotes5
BERT_DIR=/data/lixiaoya/models/bert_cased_large
MAX_LEN=200
OUTPUT_DIR=/data/lixiaoya/outputs/mrc_ner_baseline/0909/onto5_bert_tagger10_lr2e-5_drop_norm1.0_weight_warmup0.01_maxlen512
MODEL_CKPT=${OUTPUT_DIR}/epoch=25_v1.ckpt
HPARAMS_FILE=${OUTPUT_DIR}/lightning_logs/version_0/hparams.yaml
DATA_SUFFIX=.word.bmes

python3 ${REPO_PATH}/inference/tagger_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--dataset_sign ${DATA_SIGN} \
--data_file_suffix ${DATA_SUFFIX}