#!/usr/bin/env bash
# -*- coding: utf-8 -*-


TIME=0901
FILE=onto5_mrc_cased_large
REPO_PATH=/userhome/xiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
DATA_DIR=/userhome/xiaoya/dataset/new_mrc_ner/new_en_onto5
BERT_DIR=/userhome/xiaoya/bert/bert_cased_large

BERT_DROPOUT=0.2
MRC_DROPOUT=0.2
LR=2e-5
LR_MINI=3e-7
LR_SCHEDULER=polydecay
SPAN_WEIGHT=0.1
WARMUP=200
MAXLEN=210
MAXNORM=1.0
INTER_HIDDEN=2048

BATCH_SIZE=4
PREC=16
VAL_CKPT=0.2
ACC_GRAD=5
MAX_EPOCH=10
SPAN_CANDI=pred_and_gold
PROGRESS_BAR=1
WEIGHT_DECAY=0.01
OPTIM=torch.adam

OUTPUT_DIR=/userhome/xiaoya/outputs/mrc_ner/${TIME}/${FILE}_cased_large_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLEN} \
--batch_size ${BATCH_SIZE} \
--gpus="4" \
--precision=${PREC} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--lr ${LR} \
--distributed_backend=ddp \
--val_check_interval ${VAL_CKPT} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CANDI} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--max_length ${MAXLEN} \
--gradient_clip_val ${MAXNORM} \
--weight_decay ${WEIGHT_DECAY} \
--flat \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--lr_mini ${LR_MINI}


