# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: evaluate
@time: 2020/8/28 21:36
@desc: 

"""

from pytorch_lightning import Trainer

from trainer import BertLabeling

CHECKPOINT = "/mnt/mrc/train_logs/zh_msra/zh_msra_20200907_lr1e-5/epoch=14_v2.ckpt"
HPARAMS = "/mnt/mrc/train_logs/zh_msra/zh_msra_20200907_lr1e-5/lightning_logs/version_0/hparams.yaml"

# CHECKPOINT = "/mnt/mrc/train_logs/ace2004/ace2004_20200908_lr4e-5_drop0.2_bert0.1_bsz32_new_shuffle//epoch=29_v2.ckpt"
# HPARAMS = "/mnt/mrc/train_logs/ace2004/ace2004_20200908_lr4e-5_drop0.2_bert0.1_bsz32_new_shuffle/lightning_logs/version_0/hparams.yaml"


CHECKPOINT = "/mnt/mrc/train_logs/ace2004/ace2004_20200909_lr4e-5_drop0.3_bert0.1_bsz32_new_shuffle_with_impossible/epoch=19.ckpt"
HPARAMS = "//mnt/mrc/train_logs/ace2004/ace2004_20200909_lr4e-5_drop0.3_bert0.1_bsz32_new_shuffle_with_impossible/lightning_logs/version_0/hparams.yaml"


def evaluate():
    """main"""

    model = BertLabeling.load_from_checkpoint(
        checkpoint_path=CHECKPOINT,
        hparams_file=HPARAMS,
        map_location=None,
        batch_size=16,
        max_length=128,
        workers=0
    )

    trainer = Trainer(gpus=[0, 1], distributed_backend="ddp")
    trainer.test(model=model)


if __name__ == '__main__':
    evaluate()
