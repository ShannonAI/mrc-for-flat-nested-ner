#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# first create: 2020.05.05 
# last update: 2020.05.05 
# description:
# evaluate checkpoint 


REPO_PATH=/PATH-TO-REPO/mrc-for-flat-nested-ner 
config_path=${REPO_PATH}/config/zh_bert.json
data_path=/PATH-TO-BERT_MRC-DATA/zh_ontonotes4
bert_path=/PATH-TO-BERT-CHECKPOINTS/chinese_L-12_H-768_A-12
saved_model=/PATH-TO-SAVED-MODEL-CKPT/bert_finetune_model.bin
max_seq_length=100
test_batch_size=32
data_sign=zh_onto
entity_sign=flat 
n_gpu=1
seed=2333

export PYTHONPATH=${REPO_PATH}


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
