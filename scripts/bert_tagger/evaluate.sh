#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: evaluate.sh


REPO_PATH=/home/lixiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

OUTPUT_DIR=/data/lixiaoya/outputs/mrc_ner_baseline/0909/onto5_bert_tagger10_lr2e-5_drop_norm1.0_weight_warmup0.01_maxlen512
# find best checkpoint on dev in ${OUTPUT_DIR}/train_log.txt
BEST_CKPT_DEV=${OUTPUT_DIR}/epoch=25_v1.ckpt
PYTORCHLIGHT_HPARAMS=${OUTPUT_DIR}/lightning_logs/version_0/hparams.yaml
GPU_ID=0,1
MAX_LEN=220

python3 ${REPO_PATH}/evaluate/tagger_ner_evaluate.py ${BEST_CKPT_DEV} ${PYTORCHLIGHT_HPARAMS} ${GPU_ID} ${MAX_LEN}