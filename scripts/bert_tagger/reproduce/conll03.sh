#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: conll03.sh


TIME=0824
FILE=conll03_bert_tagger
REPO_PATH=/userhome/xiaoya/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
DATA_DIR=/userhome/xiaoya/dataset/tagger_ner_datasets/conll03_truecase_bmes
BERT_DIR=/userhome/xiaoya/bert/bert_cased_large

BERT_DROPOUT=0.2
LR=2e-5
LR_SCHEDULER=polydecay
MAXLEN=256
MAXNORM=1.0
DATA_SUFFIX=.word.bmes
TRAIN_BATCH_SIZE=8
GRAD_ACC=4
MAX_EPOCH=10
WEIGHT_DECAY=0.02
OPTIM=torch.adam
DATA_SIGN=en_conll03
WARMUP_PROPORTION=0.01
INTER_HIDDEN=1024

OUTPUT_DIR=/userhome/xiaoya/outputs/mrc_ner_baseline/${TIME}/${FILE}_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP_PROPORTION}_maxlen${MAXLEN}
mkdir -p $OUTPUT_DIR


CUDA_VISIBLE_DEVICES=1 python3 ${REPO_PATH}/train/bert_tagger_trainer.py \
--gpus="1" \
--progress_bar_refresh_rate 1 \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLEN} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--precision=16 \
--lr_scheduler ${LR_SCHEDULER} \
--lr ${LR} \
--val_check_interval 0.25 \
--accumulate_grad_batches ${GRAD_ACC} \
--output_dir ${OUTPUT_DIR} \
--max_epochs ${MAX_EPOCH} \
--warmup_proportion ${WARMUP_PROPORTION}  \
--max_length ${MAXLEN} \
--gradient_clip_val ${MAXNORM} \
--weight_decay ${WEIGHT_DECAY} \
--data_file_suffix ${DATA_SUFFIX} \
--optimizer ${OPTIM} \
--data_sign ${DATA_SIGN} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN}

