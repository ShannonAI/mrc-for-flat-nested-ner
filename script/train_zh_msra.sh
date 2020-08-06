#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 
# Author: Xiaoy Li


REPO_PATH=/data/xiaoya/work/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

# model params
MAX_LEN=100
LR=1e-5
START_LRATIO=1.0
END_LRATIO=1.0
SPAN_LRATIO=1.0
DP=0.3
TRAIN_BZ=8
DEV_BZ=16
TEST_BZ=16
NUM_EPOCH=15
WARMUP_PRO=-1
GradACC=1
CHECKPOINT=10
N_GPU=1
SEED=2333
DATA_SIGN=zh_msra
MODEL_SIGN=mrc-ner
ENTITY_SIGN=flat
NUM_DATA_PROCESSOR=10

# path to pretrained models and data files
BASE_DATA_DIR=/mnt
DATA_PATH=/data/xiaoya/work/datasets/mrc_ner/zh_msra
BERT_PATH=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12
CONFIG_PATH=${REPO_PATH}/config/zh_bert.json
OUTPUT_PATH=/data/xiaoya/output_mrc_ner/${DATA_SIGN}_${MAX_LEN}_${LR}_${TRAIN_BZ}_${DP}

mkdir -p ${OUTPUT_PATH}


CUDA_VISIBLE_DEVICES=2 python3 $REPO_PATH/run/train_bert_mrc.py \
--data_dir $DATA_PATH \
--num_data_processor $NUM_DATA_PROCESSOR \
--n_gpu $N_GPU \
--entity_sign $ENTITY_SIGN \
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
--gradient_accumulation_steps $GradACC


