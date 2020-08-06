#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Author: Xiaoy Li


REPO_PATH=/home/lixiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


# model params
MAX_LEN=160
LR=7e-6
START_LRATIO=1.0
END_LRATIO=1.0
SPAN_LRATIO=1.0
DP=0.2
TRAIN_BZ=12
DEV_BZ=12
TEST_BZ=16
NUM_EPOCH=6
WARMUP_PRO=-1
GradACC=1
CHECKPOINT=300
N_GPU=1
SEED=2333
DATA_SIGN=ace2004
MODEL_SIGN=mrc-ner
ENTITY_SIGN=nested
NUM_DATA_PROCESSOR=4

# path to pretrained models and data files
BASE_DATA_DIR=/data
DATA_PATH=${BASE_DATA_DIR}/datasets/ace2004
BERT_PATH=${BASE_DATA_DIR}/pretrain_ckpt/cased_L-24_H-1024_A-16
CONFIG_PATH=${REPO_PATH}/config/en_bert_large_cased.json
OUTPUT_PATH=${BASE_DATA_DIR}/output_mrc_ner/${DATA_SIGN}_${MAX_LEN}_${LR}_${TRAIN_BZ}_${DP}

mkdir -p ${OUTPUT_PATH}

CUDA_VISIBLE_DEVICES=0 python3 $REPO_PATH/run/train_bert_mrc.py \
--data_dir $DATA_PATH \
--n_gpu $N_GPU \
--entity_sign $ENTITY_SIGN \
--num_data_processor $NUM_DATA_PROCESSOR \
--data_sign $DATA_SIGN \
--bert_model $BERT_PATH \
--config_path $CONFIG_PATH \
--output_dir $OUTPUT_PATH \
--dropout $DP \
--checkpoint $CHECKPOINT \
--max_seq_length $MAX_LEN \
--train_batch_size $TRAIN_BZ \
--dev_batch_size $DEV_BZ \
--test_batch_size $TEST_BZ \
--learning_rate $LR \
--weight_start $START_LRATIO \
--weight_end $END_LRATIO \
--weight_span $SPAN_LRATIO \
--num_train_epochs $NUM_EPOCH \
--seed $SEED \
--warmup_proportion $WARMUP_PRO \
--gradient_accumulation_steps $GradACC \
--fp16