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
lr=1e-5
OPTIMIZER=adamw
INTER_HIDDEN=1536
OUTPUT_DIR=/userhome/yuxian/train_logs/zh_onto/zh_onto_${OPTIMIZER}_lr${lr}_maxlen${MAXLENGTH}_spanw${WEIGHT_SPAN}
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLENGTH \
--batch_size 8 \
--gpus="4" \
--precision=16 \
--progress_bar_refresh_rate 100 \
--lr $lr \
--workers 0 \
--distributed_backend=ddp \
--val_check_interval 0.25 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--max_epochs 10 \
--chinese \
--span_loss_candidates pred_and_gold \
--weight_span $WEIGHT_SPAN \
--mrc_dropout 0.3 \
--warmup_steps 5000 \
--gradient_clip_val 5.0 \
--final_div_factor 20 \
--classifier_intermediate_hidden_size ${INTER_HIDDEN}

