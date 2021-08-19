#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: eval.sh

REPO_PATH=/data/xiaoya/workspace/mrc-for-flat-nested-ner-github
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

OUTPUT_DIR=/data/xiaoya/outputs/mrc_ner/ace2004/debug_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen100
# find best checkpoint on dev in ${OUTPUT_DIR}/train_log.txt
BEST_CKPT_DEV=${OUTPUT_DIR}/epoch=8.ckpt
PYTORCHLIGHT_HPARAMS=${OUTPUT_DIR}/lightning_logs/version_0/hparams.yaml
GPU_ID=0,1

python3 ${REPO_PATH}/evaluate/mrc_ner_evaluate.py ${BEST_CKPT_DEV} ${PYTORCHLIGHT_HPARAMS} ${GPU_ID}