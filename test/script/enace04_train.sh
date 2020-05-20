#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# Author: Xiaoy Li 
# Description:
# train_tagger for the dataset of msra.sh

MACHINE=gpu11
EXP_ID=15_2

FOLDER_PATH=/data/xiaoya/work/gitrepo/mrc-for-flat-nested-ner
export PYTHONPATH=${FOLDER_PATH}

CONFIG_PATH=${FOLDER_PATH}/config/en_bert_base_uncased.json
DATA_PATH=/data/nfsdata2/xiaoya/data_repo/data-mrc_ner/en_ace04_short
BERT_PATH=/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12
EXPORT_DIR=/data/xiaoya/export_dir/mrc-ner
log_path=/data/xiaoya/exp-log/${DATE}


max_seq_length=150
learning_rate=4e-5
start_loss_ratio=1.0
end_loss_ratio=1.0
span_loss_ratio=1.0
dropout=0.1
train_batch_size=4
dev_batch_size=4
test_batch_size=4
max_train_expoch=10
warmup_proportion=-1
gradient_accumulation_step=1
checkpoint=600
seed=2333
n_gpu=1
data_sign=ace2004
entity_sign=nested
save_model=yes
export_model=True
model_sign=mrc
output_path=${EXPORT_DIR}/${data_sign}/${model_sign}-${max_seq_length}-${learning_rate}-${train_batch_size}-${EXP_ID}


mkdir -p ${output_path}
mkdir -p ${log_path}


CUDA_VISIBLE_DEVICES=0 python3 ${FOLDER_PATH}/run/train_bert_mrc.py \
--data_dir ${DATA_PATH} \
--n_gpu ${n_gpu} \
--dropout ${dropout} \
--entity_sign ${entity_sign} \
--data_sign ${data_sign} \
--bert_model ${BERT_PATH} \
--config_path ${CONFIG_PATH} \
--export_model ${export_model} \
--output_dir ${output_path} \
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




