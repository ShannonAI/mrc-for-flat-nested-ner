#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 

REPO_PATH=/data/xiaoya/work/gitrepo/mrc-for-flat-nested-ner
export PYTHONPATH=${REPO_PATH}

config_path=${REPO_PATH}/config/en_bert_base_uncased.json
data_path=/data/nfsdata2/xiaoya/data_repo/data-mrc_ner/en_ace04_short
bert_path=/data/nfsdata/nlp/BERT_BASE_DIR/uncased_L-12_H-768_A-12
saved_model=/data/nfsdata2/xiaoya/bert_finetune_model_0_600.bin
max_seq_length=150
test_batch_size=2
data_sign=ace2004
entity_sign=nested
n_gpu=1
seed=2333



CUDA_VISIBLE_DEVICES=2 python3 ${REPO_PATH}/run/evaluate_mrc_ner.py \
--config_path ${config_path} \
--data_dir ${data_path} \
--bert_model ${bert_path} \
--saved_model ${saved_model} \
--max_seq_length ${max_seq_length} \
--test_batch_size ${test_batch_size} \
--data_sign ${data_sign} \
--entity_sign ${entity_sign} \
--n_gpu ${n_gpu} \
--seed ${seed}