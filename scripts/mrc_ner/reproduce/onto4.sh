#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: onto4.sh
# desc: for chinese ontonotes 04

REPO_PATH=/userhome/xiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
export TOKENIZERS_PARALLELISM=false

DATA_DIR=/userhome/yuxian/data/zh_onto4
BERT_DIR=/userhome/yuxian/data/chinese_roberta_wwm_large_ext_pytorch

MAXLENGTH=128
WEIGHT_SPAN=0.1
LR=1e-5
OPTIMIZER=adamw
INTER_HIDDEN=1536
OUTPUT_DIR=/userhome/yuxian/train_logs/zh_onto/zh_onto_${OPTIMIZER}_lr${lr}_maxlen${MAXLENGTH}_spanw${WEIGHT_SPAN}
mkdir -p ${OUTPUT_DIR}

BATCH_SIZE=8
PREC=16
VAL_CKPT=0.25
ACC_GRAD=1
MAX_EPOCH=10
SPAN_CANDI=pred_and_gold
PROGRESS_BAR=1

CUDA_VISIBLE_DEVICES=0,1,2,3 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLENGTH} \
--batch_size ${BATCH_SIZE} \
--gpus="4" \
--precision=${PREC} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--lr ${LR} \
--workers 0 \
--distributed_backend=ddp \
--val_check_interval ${VAL_CKPT} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CANDI} \
--weight_span ${WEIGHT_SPAN} \
--mrc_dropout 0.3 \
--chinese \
--warmup_steps 5000 \
--gradient_clip_val 5.0 \
--final_div_factor 20 \
--classifier_intermediate_hidden_size ${INTER_HIDDEN}

