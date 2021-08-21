#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: mrc_ner_evaluate.py
# example command:
# python3 mrc_ner_evaluate.py /data/xiaoya/outputs/mrc_ner/ace2004/debug_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen100/epoch=0.ckpt \
# /data/xiaoya/outputs/mrc_ner/ace2004/debug_lr3e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen100/lightning_logs/version_2/hparams.yaml

import sys
from pytorch_lightning import Trainer
from train.mrc_ner_trainer import BertLabeling
from utils.random_seed import set_random_seed

set_random_seed(0)


def evaluate(ckpt, hparams_file, gpus=[0, 1]):
    trainer = Trainer(gpus=gpus, distributed_backend="dp")

    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=ckpt,
        hparams_file=hparams_file,
        map_location=None,
        batch_size=1,
        max_length=128,
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
        GPUS = [0, 1]

    evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS, gpus=GPUS)
