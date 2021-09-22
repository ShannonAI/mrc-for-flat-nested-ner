#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: tagger_ner_inference.py

import os
import torch
import argparse
from torch.utils.data import DataLoader
from utils.random_seed import set_random_seed
set_random_seed(0)
from train.bert_tagger_trainer import BertSequenceLabeling
from transformers import AutoTokenizer
from datasets.tagger_ner_dataset import get_labels
from datasets.tagger_ner_dataset import TaggerNERDataset
from metrics.functional.tagger_span_f1 import get_entity_from_bmes_lst, transform_predictions_to_labels


def get_dataloader(config, data_prefix="test"):
    data_path = os.path.join(config.data_dir, f"{data_prefix}{config.data_file_suffix}")
    data_tokenizer = AutoTokenizer.from_pretrained(config.bert_dir, use_fast=False, do_lower_case=config.do_lowercase)

    dataset = TaggerNERDataset(data_path, data_tokenizer, config.dataset_sign,
                               max_length=config.max_length, is_chinese=config.is_chinese, pad_to_maxlen=False)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    return dataloader, data_tokenizer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bert_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--is_chinese", action="store_true")
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--hparams_file", type=str, default="")
    parser.add_argument("--do_lowercase", action="store_true")
    parser.add_argument("--data_file_suffix", type=str, default=".word.bmes")
    parser.add_argument("--dataset_sign", type=str, choices=["en_onto", "en_conll03", "zh_onto", "zh_msra" ], default="en_onto")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    trained_tagger_ner_model = BertSequenceLabeling.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        hparams_file=args.hparams_file,
        map_location=None,
        batch_size=1,
        max_length=args.max_length,
        workers=0)

    entity_label_lst = get_labels(args.dataset_sign)
    task_idx2label = {label_idx: label_item for label_idx, label_item in enumerate(entity_label_lst)}

    data_loader, data_tokenizer = get_dataloader(args)
    vocab_path = os.path.join(args.bert_dir, "vocab.txt")
    # load token
    vocab_path = os.path.join(args.bert_dir, "vocab.txt")
    with open(vocab_path, "r") as f:
        subtokens = [token.strip() for token in f.readlines()]
    idx2tokens = {}
    for token_idx, token in enumerate(subtokens):
        idx2tokens[token_idx] = token

    for batch in data_loader:
        token_input_ids, token_type_ids, attention_mask, sequence_labels, is_wordpiece_mask = batch
        batch_size = token_input_ids.shape[0]
        logits = trained_tagger_ner_model.model(token_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_pred_lst = transform_predictions_to_labels(logits.view(batch_size, -1, len(entity_label_lst)),
                                                            is_wordpiece_mask, task_idx2label, input_type="logit")
        batch_subtokens_idx_lst = token_input_ids.numpy().tolist()[0]
        batch_subtokens_lst = [idx2tokens[item] for item in batch_subtokens_idx_lst]
        readable_input_str = data_tokenizer.decode(batch_subtokens_idx_lst, skip_special_tokens=True)

        batch_entity_lst = [get_entity_from_bmes_lst(label_lst_item) for label_lst_item in sequence_pred_lst]

        pred_entity_lst = []

        for entity_lst, subtoken_lst in zip(batch_entity_lst, batch_subtokens_lst):
            if len(entity_lst) != 0:
                # example of entity_lst:
                # ['[0,3]PER', '[6,9]ORG', '[10]PER']
                for entity_info in entity_lst:
                    if "," in entity_info:
                        inter_pos = entity_info.find(",")
                        start_pos = 1
                        end_pos = entity_info.find("]")
                        start_idx = int(entity_info[start_pos: inter_pos])
                        end_idx = int(entity_info[inter_pos+1: end_pos])
                    else:
                        start_pos = 1
                        end_pos = entity_info.find("]")
                        start_idx = int(entity_info[start_pos:end_pos])
                        end_idx = int(entity_info[start_pos:end_pos])

                    entity_tokens = subtoken_lst[start_idx: end_idx]
                    entity_string = " ".join(entity_tokens)
                    entity_string = entity_string.replace(" ##", "")
                    # append  start, end
                    pred_entity_lst.append((entity_string, entity_info[end_pos+1:]))
            else:
                pred_entity_lst.append([])

        print("*=" * 10)
        print(f"Given input: {readable_input_str}")
        print(f"Model predict: {pred_entity_lst}")



if __name__ == "__main__":
    main()
