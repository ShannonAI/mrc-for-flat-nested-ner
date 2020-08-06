#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 
# Author: Xiaoy Li


REPO_PATH=/home/lixiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


# model params
MAX_LEN=256
LR=7e-6
START_LRATIO=1.0
END_LRATIO=1.0
SPAN_LRATIO=1.0
DP=0.2
TRAIN_BZ=12
DEV_BZ=16
TEST_BZ=16
NUM_EPOCH=6
WARMUP_PRO=-1
GradACC=1
CHECKPOINT=600
N_GPU=1
SEED=2333
DATA_SIGN=zh_onto
MODEL_SIGN=mrc-ner
ENTITY_SIGN=flat
NUM_DATA_PROCESSOR=4

# path to pretrained models and data files
BASE_DATA_DIR=/data
DATA_PATH=${BASE_DATA_DIR}/mrc_ner/zh_onto4
BERT_PATH=${BASE_DATA_DIR}/pretrain_ckpt/chinese_L-12_H-768_A-12
CONFIG_PATH=${REPO_PATH}/config/zh_bert.json
OUTPUT_PATH=${BASE_DATA_DIR}/output_mrc_ner/${DATA_SIGN}_${MAX_LEN}_${LR}_${TRAIN_BZ}_${DP}

mkdir -p ${OUTPUT_PATH}


CUDA_VISIBLE_DEVICES=3 python3 $REPO_PATH/run/train_bert_mrc.py \
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