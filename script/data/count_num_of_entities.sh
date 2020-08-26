#!/usr/bin/env bash
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# count number of entities in sequence-labeling and mrc-ner datasets.
# python3 annotation_statistic.py <path_to_tagger_data_dir> <path_to_mrc_data_dir>


REPO_PATH=/data/xiaoya/work/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

TAGGER_DATADIR=$1
MRC_DATADIR=$2
NER_TYPE=$3 # nested, flat


python3 $REPO_PATH/utils/annotation_statistic.py $NER_TYPE $TAGGER_DATADIR $MRC_DATADIR
