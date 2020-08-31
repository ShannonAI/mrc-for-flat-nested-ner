#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoy li
# file: eval_ckpt.py


REPO_PATH=/home/lixiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
FILE_NAME=eval_ckpt


# model params
MAX_LEN=260
START_LRATIO=1.0
END_LRATIO=1.0
SPAN_LRATIO=1.0
ENTITY_THRESHOLD=0.15
DP=0.1
TEST_BZ=12
N_GPU=1
SEED=2333
DATA_SIGN=zh_onto
MODEL_SIGN=mrc-ner
ENTITY_SIGN=flat
NUM_DATA_PROCESSOR=1


DATA_BASE_DIR=/mnt
SAVED_MODEL_DIR=${DATA_BASE_DIR}/output_mrc_ner/zh_onto_200_1e-5_12_0.1_zhonto2
MODEL_FILE=bert_finetune_model_0_3600.bin
LOG_FILE=eval_test_log.txt
DATA_PATH=${DATA_BASE_DIR}/mrc_ner_data/zh_onto4
BERT_PATH=${DATA_BASE_DIR}/pretrain_ckpt/chinese_L-12_H-768_A-12
CONFIG_PATH=${REPO_PATH}/config/zh_bert.json
EVAL_MODEL_PATH=${SAVED_MODEL_DIR}/${MODEL_FILE}
LOG_FILE_PATH=${SAVED_MODEL_DIR}/${LOG_FILE}


CUDA_VISIBLE_DEVICES=0  python3 $REPO_PATH/run/evaluate_mrc_ner.py \
--config_path $CONFIG_PATH \
--data_dir $DATA_PATH \
--bert_model $BERT_PATH \
--saved_model $EVAL_MODEL_PATH \
--logfile_path $LOG_FILE_PATH \
--data_sign $DATA_SIGN \
--entity_sign $ENTITY_SIGN \
--num_data_processor $NUM_DATA_PROCESSOR \
--max_seq_length $MAX_LEN \
--test_batch_size $TEST_BZ \
--dropout $DP \
--weight_start $START_LRATIO \
--weight_end $END_LRATIO \
--weight_span $SPAN_LRATIO \
--entity_threshold $ENTITY_THRESHOLD \
--seed $SEED \
--n_gpu $N_GPU

# sleep 1 && tail -f ${LOG_FILE_PATH}

