# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: trainer
@time: 2020/9/6 14:26
@desc: pytorch-lightning trainer

"""


import argparse
from typing import List, Dict, Union
import os

import pytorch_lightning as pl
import torch
from torch import Tensor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import BertConfig, AdamW
from tokenizers import BertWordPieceTokenizer
from collections import namedtuple
from models.query_ner_config import BertQueryNerConfig

from models.bert_query_ner import BertQueryNER
from datasets.mrc_ner_dataset import MRCNERDataset
from datasets.truncate_dataset import TruncateDataset
from utils.get_parser import get_parser
from metrics.functional.query_span_f1 import query_span_f1
from utils.radom_seed import set_random_seed


set_random_seed(0)


class BertLabeling(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir

        bert_config = BertQueryNerConfig.from_pretrained(args.bert_config_dir,
                                                         mrc_dropout=args.mrc_dropout)

        self.model = BertQueryNER.from_pretrained(args.bert_config_dir,
                                                  config=bert_config)
        print(self.model)
        self.ce_loss = CrossEntropyLoss(reduction="none")
        self.bce_loss = BCEWithLogitsLoss(reduction="none")
        # todo(yuxian): 由于match loss是n^2的，应该特殊调整一下loss rate
        self.loss_wb = args.weight_start
        self.loss_we = args.weight_end
        self.loss_ws = args.weight_span

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mrc_dropout", type=float, default=0.1,
                            help="mrc dropout rate")
        parser.add_argument("--weight_start", type=float, default=1.0)
        parser.add_argument("--weight_end", type=float, default=1.0)
        parser.add_argument("--weight_span", type=float, default=1.0)

        return parser

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.args.lr,
                          eps=self.args.adam_epsilon)
        num_gpus = len(str(self.args.gpus).split(","))
        t_total = len(self.train_dataloader()) * self.args.max_epochs // self.args.accumulate_grad_batches // num_gpus
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        """"""
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, pred: torch.Tensor, label: torch.Tensor, label_mask: torch.Tensor):
        """
        compute mask clf loss
        Args:
            pred: [batch, seq_length, num_labels]
            label: [batch, seq_length]
            label_mask: [batch, seq_length] 1 if compute loss at this position, else 0

        Returns:

        """
        epsilon = 1e-10
        batch_size, seq_length = label.shape
        loss = self.loss_fn(pred.view(batch_size * seq_length, -1), label.view(-1))
        label_mask = label_mask.float().view(-1)
        loss *= label_mask
        loss = loss.sum() / (label_mask.sum() + epsilon)
        return loss

    @staticmethod
    def get_multiclass_f1(confusion_matrix: torch.Tensor, label_idxs: List[int] = None):
        """
        get multicalss f1 score according to confusion_matrix
        Args:
            confusion_matrix: [num_labels, num_labels], where each entry C_{i,j} is the number of observations
                              in group i that were predicted in group j.
            label_idxs: calculate f1 on those labels
        Returns:
            labels_f1: [len(lbael_idxs)]
        """

        label_idxs = label_idxs or list(range(confusion_matrix.shape[0]))

        epsilon = 1e-10

        label_f1s = []
        for idx in label_idxs:
            tp = confusion_matrix[idx, idx]
            tp_and_fp = confusion_matrix[:, idx].sum()
            tp_and_fn = confusion_matrix[idx, :].sum()
            precision = tp / (tp_and_fp + epsilon)
            recall = tp / (tp_and_fn + epsilon)
            f1 = precision * recall * 2 / (precision + recall + epsilon)
            label_f1s.append(f1)
        return torch.stack(label_f1s)

    def training_step(self, batch, batch_idx):
        """"""
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        tokens, token_type_ids, start_labels, end_labels, label_mask, match_labels = batch
        batch_size, seq_len = tokens.size()
        float_label_mask = label_mask.float().view(-1)

        match_label_row_mask = label_mask.bool().unsqueeze(-1).repeat([1, 1, seq_len])
        match_label_col_mask = label_mask.bool().unsqueeze(-2).repeat([1, seq_len, 1])
        match_label_mask = match_label_row_mask & match_label_col_mask
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()

        # num_tasks * [bsz, length, num_labels]
        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)

        start_loss = self.ce_loss(start_logits.view(-1, 2), start_labels.view(-1))
        start_loss = (start_loss * float_label_mask).sum() / float_label_mask.sum()
        end_loss = self.ce_loss(end_logits.view(-1, 2), end_labels.view(-1))
        end_loss = (end_loss * float_label_mask).sum() / float_label_mask.sum()
        match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = (match_loss * float_match_label_mask).sum() / float_match_label_mask.sum()

        total_loss = self.loss_wb * start_loss + self.loss_we * end_loss + self.loss_ws * match_loss

        tf_board_logs[f"train_loss"] = total_loss
        tf_board_logs[f"start_loss"] = start_loss
        tf_board_logs[f"end_loss"] = end_loss
        tf_board_logs[f"match_loss"] = match_loss

        return {'loss': total_loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """todo(yuxian): add span f1"""

        output = {}

        tokens, token_type_ids, start_labels, end_labels, label_mask, match_labels = batch
        batch_size, seq_len = tokens.size()
        float_label_mask = label_mask.float().view(-1)

        match_label_row_mask = label_mask.bool().unsqueeze(-1).repeat([1, 1, seq_len])
        match_label_col_mask = label_mask.bool().unsqueeze(-2).repeat([1, seq_len, 1])
        match_label_mask = match_label_row_mask & match_label_col_mask
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()

        # num_tasks * [bsz, length, num_labels]
        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)

        start_loss = self.ce_loss(start_logits.view(-1, 2), start_labels.view(-1))
        start_loss = (start_loss * float_label_mask).sum() / float_label_mask.sum()
        end_loss = self.ce_loss(end_logits.view(-1, 2), end_labels.view(-1))
        end_loss = (end_loss * float_label_mask).sum() / float_label_mask.sum()
        match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = (match_loss * float_match_label_mask).sum() / float_match_label_mask.sum()

        total_loss = self.loss_wb * start_loss + self.loss_we * end_loss + self.loss_ws * match_loss

        output[f"val_loss"] = total_loss
        output[f"start_loss"] = start_loss
        output[f"end_loss"] = end_loss
        output[f"match_loss"] = match_loss

        span_f1_stats = query_span_f1(start_logits=start_logits, end_logits=end_logits, match_logits=span_logits,
                                      label_mask=label_mask, match_labels=match_labels)
        output["span_f1_stats"] = span_f1_stats

        return output

    def validation_epoch_end(self, outputs):
        """"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"span_precision"] = span_precision
        tensorboard_logs[f"span_recall"] = span_recall
        tensorboard_logs[f"span_f1"] = span_f1

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(
        self,
        outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        """"""
        return self.validation_epoch_end(outputs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")
        # return self.get_dataloader("dev", 100)

    def val_dataloader(self):
        return self.get_dataloader("dev")
        # return self.get_dataloader("dev", 100)

    def test_dataloader(self):
        return self.get_dataloader("test")

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """get training dataloader"""
        """
        load_mmap_dataset
        """
        json_path = os.path.join(self.data_dir, f"mrc-ner.{prefix}")
        vocab_path = os.path.join(self.bert_dir, "vocab.txt")
        dataset = MRCNERDataset(json_path=json_path,
                                tokenizer=BertWordPieceTokenizer(vocab_file=vocab_path),
                                max_length=self.args.max_length)

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

        return dataloader


def run_dataloader():
    """test dataloader"""
    parser = get_parser()

    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    args.workers = 0

    model = BertLabeling(args)
    from tokenizers import BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_config_dir, "vocab.txt"))

    loader = model.get_dataloader("dev", limit=1000)
    for d in loader:
        input_ids = d[0][0].tolist()
        match_labels = d[-1][0]
        start_positions, end_positions = torch.where(match_labels > 0)
        start_positions = start_positions.tolist()
        end_positions = end_positions.tolist()
        if not start_positions:
            continue
        print("="*20)
        print(tokenizer.decode(input_ids, skip_special_tokens=False))
        for start, end in zip(start_positions, end_positions):
            print(tokenizer.decode(input_ids[start: end+1]))


def main():
    """main"""
    parser = get_parser()

    # add model specific args
    parser = BertLabeling.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    model = BertLabeling(args)
    checkpoint_callback = ModelCheckpoint(
        filepath=args.default_root_dir,
        save_top_k=10,
        verbose=True,
        monitor="span_f1",
        period=-1,
        mode="auto",
    )
    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model)


if __name__ == '__main__':
    # run_dataloader()
    main()
