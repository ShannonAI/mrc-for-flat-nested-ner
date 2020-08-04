#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 
# Author: Xiaoy Li 
# On One 12G TITAN XP

EXP-ID=22_1
FOLDER_PATH=/data/xiaoya/work/mrc-for-flat-nested-ner
DATA_PATH=/data/xiaoya/work/datasets/mrc_ner/zh_onto4
BERT_PATH=/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12
EXPORT_DIR=/data/xiaoya/output_mrc_ner
CONFIG_PATH=${FOLDER_PATH}/config/zh_bert.json


max_seq_length=50
learning_rate=8e-6
start_loss_ratio=1.0
end_loss_ratio=1.0
span_loss_ratio=1.0
dropout=0.2
train_batch_size=12
dev_batch_size=32
test_batch_size=32
max_train_expoch=6
warmup_proportion=-1
gradient_accumulation_step=1
checkpoint=600
n_gpu=1
seed=2333
data_sign=zh_onto 
export_model=True
model_sign=mrc-ner
data_cache=True
output_path=${EXPORT_DIR}/${data_sign}/${model_sign}-${data_sign}-${EXP_ID}-${max_seq_length}-${learning_rate}-${train_batch_size}-${dropout}


mkdir -p ${output_path}
export PYTHONPATH=${FOLDER_PATH}


CUDA_VISIBLE_DEVICES=2 python3 ${FOLDER_PATH}/run/train_bert_mrc.py \
--data_dir ${DATA_PATH} \
--n_gpu ${n_gpu} \
--entity_sign flat \
--data_sign ${data_sign} \
--bert_model ${BERT_PATH} \
--config_path ${CONFIG_PATH} \
--export_model ${export_model} \
--output_dir ${output_path} \
--dropout ${dropout} \
--checkpoint ${checkpoint} \
--max_seq_length ${max_seq_length} \
--train_batch_size ${train_batch_size} \
--dev_batch_size ${dev_batch_size} \
--test_batch_size ${test_batch_size} \
--learning_rate ${learning_rate} \
--weight_start ${start_loss_ratio} \
--weight_end ${end_loss_ratio} \
--weight_span ${span_loss_ratio} \
--num_train_epochs ${max_train_expoch} \
--seed ${seed} \
--warmup_proportion ${warmup_proportion} \
--gradient_accumulation_steps ${gradient_accumulation_step}




