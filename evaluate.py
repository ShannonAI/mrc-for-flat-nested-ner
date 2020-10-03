# encoding: utf-8


import os
from pytorch_lightning import Trainer

from trainer import BertLabeling


def evaluate(ckpt, hparams_file):
    """main"""

    trainer = Trainer(gpus=[0, 1], distributed_backend="ddp")

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
    # ace04
    HPARAMS = "/mnt/mrc/train_logs/ace2004/ace2004_20200911reproduce_epoch15_lr3e-5_drop0.3_norm1.0_bsz32_hard_span_weight0.1_warmup0_maxlen128_newtrunc_debug/lightning_logs/version_0/hparams.yaml"
    CHECKPOINTS = "/mnt/mrc/train_logs/ace2004/ace2004_20200911reproduce_epoch15_lr3e-5_drop0.3_norm1.0_bsz32_hard_span_weight0.1_warmup0_maxlen128_newtrunc_debug/epoch=10_v0.ckpt"
    # DIR = "/mnt/mrc/train_logs/ace2004/ace2004_20200910_lr3e-5_drop0.3_bert0.1_bsz32_hard_loss_bce_weight_span0.05"
    # CHECKPOINTS = [os.path.join(DIR, x) for x in os.listdir(DIR)]

    # ace04-large
    HPARAMS = "/mnt/mrc/train_logs/ace2004/ace2004_20200910reproduce_lr3e-5_drop0.3_norm1.0_bsz32_hard_span_weight0.1_warmup0_maxlen128_newtrunc_debug/lightning_logs/version_2/hparams.yaml"
    CHECKPOINTS = "/mnt/mrc/train_logs/ace2004/ace2004_20200910reproduce_lr3e-5_drop0.3_norm1.0_bsz32_hard_span_weight0.1_warmup0_maxlen128_newtrunc_debug/epoch=10.ckpt"

    # ace05
    # HPARAMS = "/mnt/mrc/train_logs/ace2005/ace2005_20200911_lr3e-5_drop0.3_norm1.0_bsz32_hard_span_weight0.1_warmup0_maxlen128_newtrunc_debug/lightning_logs/version_0/hparams.yaml"
    # CHECKPOINTS = "/mnt/mrc/train_logs/ace2005/ace2005_20200911_lr3e-5_drop0.3_norm1.0_bsz32_hard_span_weight0.1_warmup0_maxlen128_newtrunc_debug/epoch=15.ckpt"

    # zh_msra
    CHECKPOINTS = "/mnt/mrc/train_logs/zh_msra/zh_msra_20200911_for_flat_debug/epoch=2_v1.ckpt"
    HPARAMS = "/mnt/mrc/train_logs/zh_msra/zh_msra_20200911_for_flat_debug/lightning_logs/version_2/hparams.yaml"


    evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)
