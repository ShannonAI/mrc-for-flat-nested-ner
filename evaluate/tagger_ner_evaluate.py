#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: tagger_ner_evaluate.py
# example command:


import sys
from pytorch_lightning import Trainer
from train.bert_tagger_trainer import BertSequenceLabeling
from utils.random_seed import set_random_seed

set_random_seed(0)


def evaluate(ckpt, hparams_file, gpus=[0, 1], max_length=300):
    trainer = Trainer(gpus=gpus, distributed_backend="dp")

    model = BertSequenceLabeling.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=1,
        max_length=max_length,
        workers=0
    )
    trainer.test(model=model)


if __name__ == '__main__':
    # example of running evaluate.py
    # CHECKPOINTS = "/mnt/mrc/train_logs/zh_msra/zh_msra_20200911_for_flat_debug/epoch=2_v1.ckpt"
    # HPARAMS = "/mnt/mrc/train_logs/zh_msra/zh_msra_20200911_for_flat_debug/lightning_logs/version_2/hparams.yaml"
    # GPUS="1,2,3"
    CHECKPOINTS = sys.argv[1]
    HPARAMS = sys.argv[2]

    try:
        GPUS = [int(gpu_item) for gpu_item in sys.argv[3].strip().split(",")]
    except:
        GPUS = [0]

    try:
        MAXLEN = int(sys.argv[4])
    except:
        MAXLEN = 300

    evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS, gpus=GPUS, max_length=MAXLEN, )
