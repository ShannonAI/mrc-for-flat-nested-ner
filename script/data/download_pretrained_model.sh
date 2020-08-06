#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# download pretrained model ckpt



BERT_PRETRAIN_CKPT=$1
MODEL_NAME=$2



if [[ $MODEL_NAME == "en_bert_cb" ]]; then
    mkdir -p $BERT_PRETRAIN_CKPT
    echo "DownLoad English BERT-Base, Cased"
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/cased_L-12_H-768_A-12.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/cased_L-12_H-768_A-12.zip
elif [[ $MODEL_NAME == "en_bert_cl" ]]; then
    echo "DownLoad English BERT-Large, Cased"
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/cased_L-24_H-1024_A-16.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/cased_L-24_H-1024_A-16.zip
elif [[ $MODEL_NAME == "en_bert_ucb" ]]; then
    echo "DownLoad English BERT-Base, Uncased"
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/uncased_L-12_H-768_A-12.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/uncased_L-12_H-768_A-12.zip
elif [[ $MODEL_NAME == "en_bert_ucl" ]]; then
    echo "DownLoad English BERT-Large, Uncased"
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/uncased_L-24_H-1024_A-16.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/uncased_L-24_H-1024_A-16.zip
elif [[ $MODEL_NAME == "en_bert_wwm_cl" ]]; then
    echo "DownLoad English BERT-Large, Cased (Whole Word Masking)"
    wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/wwm_cased_L-24_H-1024_A-16.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/wwm_cased_L-24_H-1024_A-16.zip
elif [[ $MODEL_NAME == "en_bert_wwm_ucl" ]]; then
    echo "DownLoad English BERT-Large, Uncased (Whole Word Masking)"
    wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/wwm_uncased_L-24_H-1024_A-16.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/wwm_uncased_L-24_H-1024_A-16.zip
elif [[ $MODEL_NAME == "en_spanbert_cb" ]]; then
    echo "DownLoad English SpanBERT-Base, Cased"
    wget https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/spanbert_hf_base.tar.gz -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/spanbert_hf_base.tar.gz
elif [[ $MODEL_NAME == "en_spanbert_cl" ]]; then
    echo "DownLoad English SpanBERT-Large, Cased"
    wget https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf.tar.gz -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/spanbert_hf.tar.gz -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/spanbert_hf.tar.gz
elif [[ $MODEL_NAME == "en_bert_tiny" ]]; then
    each "DownLoad English BERT-Tiny, Uncased"
    wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/uncased_L-2_H-128_A-2.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/uncased_L-2_H-128_A-2.zip
elif [[ $MODEL_NAME == "zh_bert" ]]; then
    each "DownLoad Chinese BERT-Base"
    wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip -P $BERT_PRETRAIN_CKPT
    unzip $BERT_PRETRAIN_CKPT/chinese_L-12_H-768_A-12.zip -d $BERT_PRETRAIN_CKPT
    rm $BERT_PRETRAIN_CKPT/chinese_L-12_H-768_A-12.zip
else
    echo 'Unknown argment 2 (Model Sign)'
fi