#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: ace04.sh

TIME=0910
REPO_PATH=/userhome/xiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
DATA_DIR=/userhome/xiaoya/dataset/ace2004
BERT_DIR=/userhome/xiaoya/bert/bert_uncased_large

BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=3e-5
SPAN_WEIGHT=0.1
WARMUP=0
MAXLEN=128
MAXNORM=1.0
INTER_HIDDEN=2048

BATCH_SIZE=4
PREC=16
VAL_CKPT=0.25
ACC_GRAD=2
MAX_EPOCH=20
SPAN_CANDI=pred_and_gold
PROGRESS_BAR=1

OUTPUT_DIR=/userhome/xiaoya/outputs/mrc-ner/${TIME}/ace2004/large_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--gpus="2" \
--distributed_backend=ddp \
--workers 0 \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLEN} \
--batch_size ${BATCH_SIZE} \
--precision=${PREC} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--lr ${LR} \
--val_check_interval ${VAL_CKPT} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT}\
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CANDI} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--gradient_clip_val ${MAXNORM} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN}

