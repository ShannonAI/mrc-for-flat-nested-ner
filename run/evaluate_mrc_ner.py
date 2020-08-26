#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# author: xiaoy li
# description:
# evaluate


import os
import random
import argparse
import numpy as np


import torch
from data_loader.model_config import Config
from data_loader.mrc_data_processor import Conll03Processor, MSRAProcessor, Onto4ZhProcessor, Onto5EngProcessor, GeniaProcessor, ACE2004Processor, ACE2005Processor, ResumeZhProcessor
from data_loader.mrc_data_loader import MRCNERDataLoader
from model.bert_mrc import BertQueryNER
from utils.get_logger import logger_to_file
from data_loader.bert_tokenizer import BertTokenizer4Tagger
from metric.mrc_ner_evaluate  import flat_ner_performance, nested_ner_performance



def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", default="/home/lixiaoya/", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default=None, type=str, )
    parser.add_argument("--saved_model", type=str, default="/home/lixiaoya")
    parser.add_argument("--logfile_path", type=str, default="/home/lixiaoya/eval_log.txt")

    parser.add_argument("--data_sign", type=str, default="msra_ner")
    parser.add_argument("--do_lower_case", default=False, action='store_true', help="lower case of input tokens.")
    parser.add_argument("--entity_sign", type=str, default="flat")
    parser.add_argument("--num_data_processor", default=1, type=int, help="number of data processor.")
    parser.add_argument("--data_cache", default=True, action='store_false')

    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--weight_start", type=float, default=1.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_span", type=float, default=1.0)
    parser.add_argument("--entity_threshold", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--n_gpu", type=int, default=1)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.cuda.manual_seed_all(args.seed)

    return args


def load_data(config, logger):

    logger.info("-*-"*10)
    logger.info(f"current data_sign: {config.data_sign}")

    if config.data_sign == "conll03":
        data_processor = Conll03Processor()
    elif config.data_sign == "zh_msra":
        data_processor = MSRAProcessor()
    elif config.data_sign == "zh_onto":
        data_processor = Onto4ZhProcessor()
    elif config.data_sign == "en_onto":
        data_processor = Onto5EngProcessor()
    elif config.data_sign == "genia":
        data_processor = GeniaProcessor()
    elif config.data_sign == "ace2004":
        data_processor = ACE2004Processor()
    elif config.data_sign == "ace2005":
        data_processor = ACE2005Processor()
    elif config.data_sign == "resume":
        data_processor = ResumeZhProcessor()
    else:
        raise ValueError("Please Notice that your data_sign DO NOT exits !!!!!")


    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer4Tagger.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    dataset_loaders = MRCNERDataLoader(config, data_processor, label_list, tokenizer, mode="test", allow_impossible=True)
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test", num_data_processor=config.num_data_processor, logger=logger)

    return test_dataloader, label_list


def load_model(config, label_list):
    device = torch.device("cuda")
    n_gpu = config.n_gpu
    model = BertQueryNER(config)
    checkpoint = torch.load(config.saved_model)
    model.load_state_dict(checkpoint)
    model.to(device)
    if config.n_gpu >1 :
        model = torch.nn.DataParallel(model)

    return model, device, n_gpu



def eval_checkpoint(model_object, eval_dataloader, config, \
    device, n_gpu, label_list, logger, eval_sign="test"):

    logger.info("$="*20)
    logger.info(f"EVAL {config.saved_model} on Test Set. ")
    model_object.eval()

    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    mask_lst = []

    start_gold_lst = []
    end_gold_lst = []
    span_gold_lst = []

    ner_cate_lst = []


    for eval_idx, eval_batch in enumerate(eval_dataloader):
        input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, span_label_mask, ner_cate = eval_batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
        span_pos = span_pos.to(device)

        with torch.no_grad():
            start_logits, end_logits, span_logits = model_object(input_ids, segment_ids, input_mask)

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        span_pos = span_pos.to("cpu").numpy().tolist()

        start_label = start_logits.detach().cpu().numpy().tolist()
        end_label = end_logits.detach().cpu().numpy().tolist()
        span_label = span_logits.detach().cpu().numpy().tolist()

        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        ner_cate_lst += ner_cate.numpy().tolist()
        mask_lst += input_mask

        start_pred_lst += start_label
        end_pred_lst += end_label
        span_pred_lst += span_label

        start_gold_lst += start_pos
        end_gold_lst += end_pos
        span_gold_lst += span_pos


    if config.entity_sign == "flat":
        acc, pre, rec, f1 = flat_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)
    else:
        acc, pre, rec, f1 = nested_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)


    logger.info("=*="*10)
    logger.info("eval on test set : acc, pre, rec, f1")
    logger.info(f"{acc}, {pre}, {rec}, {f1}")

    return acc, pre, rec, f1


def merge_config(args_config, logger=None):
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config(logger=logger)
    return model_config


def main():
    args_config = args_parser()
    logger = logger_to_file(args_config.logfile_path)
    config = merge_config(args_config, logger=logger)
    test_loader, label_list = load_data(config, logger)
    model, device, n_gpu = load_model(config, label_list, )
    acc, pre, rec, f1 = eval_checkpoint(model, test_loader, config, device, n_gpu, label_list, logger, eval_sign="test")


if __name__ == "__main__":
    main()





