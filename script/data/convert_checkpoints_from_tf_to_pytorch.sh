#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# convert model checkpoints from tensorflow to pytorch
# reference to: pytorch-pretrained-bert 0.6.1


REPO_PATH=/home/lixiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

# model dir name
MODEL=$1
if [[ $MODEL == "zh_bert" ]]; then
  MODEL_DIR=chinese_L-12_H-768_A-12
elif [[ $MODEL == "en_bert_cl" ]]; then
  MODEL_DIR=cased_L-24_H-1024_A-16
elif [[ $MODEL == "en_bert_ucl" ]]; then
  MODEL_DIR=uncased_L-24_H-1024_A-16
else
    echo 'Unknown ARG1 (Model Sign)'
fi
echo "Model DIR NAME IS: ${MODEL_DIR}"


# data_path
BASE_DIR=/data/pretrain_ckpt
BERT_DIR=${BASE_DIR}/${MODEL_DIR}
BERT_CONFIG=${BERT_DIR}/bert_config.json
TF_CKPT_PATH=${BERT_DIR}/bert_model.ckpt
PYTORCH_CKPT_PATH=${BERT_DIR}/pytorch_model.bin


python3 $REPO_PATH/utils/convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path $TF_CKPT_PATH \
--bert_config_file $BERT_CONFIG \
--pytorch_dump_path $PYTORCH_CKPT_PATH