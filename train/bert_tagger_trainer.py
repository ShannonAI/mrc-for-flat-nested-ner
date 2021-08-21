#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bert_tagger_trainer.py

import os
import re
import argparse
import logging
from typing import Dict
from collections import namedtuple
from utils.random_seed import set_random_seed
set_random_seed(0)

import torch
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn.modules import CrossEntropyLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from utils.get_parser import get_parser
from datasets.tagger_ner_dataset import get_labels, TaggerNERDataset
from datasets.truncate_dataset import TruncateDataset
from datasets.collate_functions import tagger_collate_to_max_length
from metrics.tagger_span_f1 import TaggerSpanF1
from metrics.functional.tagger_span_f1 import transform_predictions_to_labels
from models.bert_tagger import BertTagger
from models.model_config import BertTaggerConfig


class BertSequenceLabeling(pl.LightningModule):
    def __init__(
        self,
        args: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        format = '%(asctime)s - %(name)s - %(message)s'
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
            logging.basicConfig(format=format, filename=os.path.join(self.args.output_dir, "eval_result_log.txt"), level=logging.INFO)
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)
            logging.basicConfig(format=format, filename=os.path.join(self.args.output_dir, "eval_test.txt"), level=logging.INFO)

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir
        self.task_labels = get_labels(self.args.data_sign)
        self.num_labels = len(self.task_labels)
        self.task_idx2label = {label_idx : label_item for label_idx, label_item in enumerate(get_labels(self.args.data_sign))}
        bert_config = BertTaggerConfig.from_pretrained(args.bert_config_dir,
                                                       hidden_dropout_prob=args.bert_dropout,
                                                       attention_probs_dropout_prob=args.bert_dropout,
                                                       classifier_dropout=args.classifier_dropout,
                                                       num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_config_dir, use_fast=False, do_lower_case=args.do_lowercase)
        self.model = BertTagger.from_pretrained(args.bert_config_dir, config=bert_config)
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.loss_func = CrossEntropyLoss()
        self.span_f1 = TaggerSpanF1()
        self.chinese = args.chinese
        self.optimizer = args.optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train_batch_size", type=int, default=8, help="batch size")
        parser.add_argument("--eval_batch_size", type=int, default=8, help="batch size")
        parser.add_argument("--bert_dropout", type=float, default=0.1, help="bert dropout rate")
        parser.add_argument("--classifier_dropout", type=float, default=0.1)
        parser.add_argument("--chinese", action="store_true", help="is chinese dataset")
        parser.add_argument("--optimizer", choices=["adamw", "torch.adam"], default="adamw", help="loss type")
        parser.add_argument("--final_div_factor", type=float, default=1e4, help="final div factor of linear decay scheduler")
        parser.add_argument("--output_dir", type=str, default="", help="the path for saving intermediate model checkpoints.")
        parser.add_argument("--lr_scheduler", type=str, default="linear_decay", help="lr scheduler")
        parser.add_argument("--data_sign", type=str, default="en_conll03", help="data signature for the dataset.")
        parser.add_argument("--polydecay_ratio", type=float, default=4, help="ratio for polydecay learing rate scheduler.")
        parser.add_argument("--do_lowercase", action="store_true", )
        parser.add_argument("--data_file_suffix", type=str, default=".char.bmes")
        parser.add_argument("--lr_scheulder", type=str, default="polydecay")
        parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")

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
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        elif self.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            raise ValueError("Optimizer type does not exist.")
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(warmup_steps/t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear')
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif self.args.lr_scheulder == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total, lr_end=self.args.lr / self.args.polydecay_ratio)
        else:
            raise ValueError
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    def compute_loss(self, sequence_logits, sequence_labels, input_mask=None):
        if input_mask is not None:
            active_loss = input_mask.view(-1) == 1
            active_logits = sequence_logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, sequence_labels.view(-1), torch.tensor(self.loss_func.ignore_index).type_as(sequence_labels)
            )
            loss = self.loss_func(active_logits, active_labels)
        else:
            loss = self.loss_func(sequence_logits.view(-1, self.num_labels), sequence_labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        tf_board_logs = {"lr": self.trainer.optimizers[0].param_groups[0]['lr']}
        token_input_ids, token_type_ids, attention_mask, sequence_labels, is_wordpiece_mask = batch

        logits = self.model(token_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = self.compute_loss(logits, sequence_labels, input_mask=attention_mask)
        tf_board_logs[f"train_loss"] = loss

        return {'loss': loss, 'log': tf_board_logs}

    def validation_step(self, batch, batch_idx):
        output = {}

        token_input_ids, token_type_ids, attention_mask, sequence_labels, is_wordpiece_mask = batch
        batch_size = token_input_ids.shape[0]
        print(batch_size)
        logits = self.model(token_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = self.compute_loss(logits, sequence_labels, input_mask=attention_mask)
        output[f"val_loss"] = loss

        sequence_pred_lst = transform_predictions_to_labels(logits.view(batch_size, -1, len(self.task_labels)), is_wordpiece_mask, self.task_idx2label, input_type="logit")
        sequence_gold_lst = transform_predictions_to_labels(sequence_labels, is_wordpiece_mask, self.task_idx2label, input_type="label")
        span_f1_stats = self.span_f1(sequence_pred_lst, sequence_gold_lst)
        output["span_f1_stats"] = span_f1_stats

        return output

    def validation_epoch_end(self, outputs):
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
        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> valid_f1 is: {span_f1}")

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """"""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> Dict[str, Dict[str, Tensor]]:
        return self.validation_epoch_end(outputs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test")

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:
        """get train/dev/test dataloader"""
        data_path = os.path.join(self.data_dir, f"{prefix}{self.args.data_file_suffix}")
        dataset = TaggerNERDataset(data_path, self.tokenizer, self.args.data_sign,
                                   max_length=self.args.max_length, is_chinese=self.args.chinese,
                                   pad_to_maxlen=False)

        if limit is not None:
            dataset = TruncateDataset(dataset, limit)

        if prefix == "train":
            batch_size = self.args.train_batch_size
            # define data_generator will help experiment reproducibility.
            # cannot use random data sampler since the gradient may explode.
            data_generator = torch.Generator()
            data_generator.manual_seed(self.args.seed)
            data_sampler = RandomSampler(dataset, generator=data_generator)
        else:
            data_sampler = SequentialSampler(dataset)
            batch_size = self.args.eval_batch_size

        dataloader = DataLoader(
            dataset=dataset, sampler=data_sampler,
            batch_size=batch_size, num_workers=self.args.workers,
            collate_fn=tagger_collate_to_max_length
        )

        return dataloader


def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt", only_keep_the_best_ckpt: bool = True):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN = re.compile(r"val_f1 reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
    best_f1_on_dev = 0
    best_checkpoint_on_dev = ""
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(
            re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("val_f1 reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(
            " as top", "")

        if current_f1 >= best_f1_on_dev:
            if only_keep_the_best_ckpt and len(best_checkpoint_on_dev) != 0:
                os.remove(best_checkpoint_on_dev)
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def main():
    """main"""
    parser = get_parser()

    # add model specific args
    parser = BertSequenceLabeling.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = BertSequenceLabeling(args)
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])
    checkpoint_callback = ModelCheckpoint(
        filepath=args.output_dir,
        save_top_k=20,
        verbose=True,
        monitor="span_f1",
        period=-1,
        mode="max",
    )
    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        deterministic=True
    )

    trainer.fit(model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.output_dir,)
    trainer.result_logger.info("=&" * 20)
    trainer.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    trainer.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    checkpoint = torch.load(path_to_best_checkpoint)
    trainer.load_state_dict(checkpoint['state_dict'])
    trainer.test(trainer)
    trainer.result_logger.info("=&" * 20)


if __name__ == '__main__':
    main()
