#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Description:
# run_machine_comprehension.py 
# Please Notice that the data should contain 
# multi answers 
# need pay MORE attention when loading data 



import os 
import argparse 
import numpy as np 
import random
import torch

from data_loader.model_config import Config 
from data_loader.mrc_data_loader import MRCNERDataLoader
from data_loader.mrc_data_processor import Conll03Processor, MSRAProcessor, Onto4ZhProcessor, Onto5EngProcessor, GeniaProcessor, ACE2004Processor, ACE2005Processor, ResumeZhProcessor
from layer.optim import AdamW, lr_linear_decay, BertAdam
from model.bert_mrc import BertQueryNER
from data_loader.bert_tokenizer import BertTokenizer4Tagger 
from metric.mrc_ner_evaluate  import flat_ner_performance, nested_ner_performance



def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()

    # requires parameters 
    parser.add_argument("--config_path", default="/home/lixiaoya/", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default=None, type=str,)
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--output_dir", type=str, default="/home/lixiaoya/output")
    parser.add_argument("--data_sign", type=str, default="msra_ner")
    parser.add_argument("--weight_start", type=float, default=1.0) 
    parser.add_argument("--weight_end", type=float, default=1.0) 
    parser.add_argument("--weight_span", type=float, default=1.0) 
    parser.add_argument("--entity_sign", type=str, default="flat")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--entity_threshold", type=float, default=0.5)
    parser.add_argument("--num_data_processor", default=1, type=int, help="number of data processor.")
    parser.add_argument("--data_cache", default=True, action='store_false')
    parser.add_argument("--export_model", default=True, action='store_false')
    parser.add_argument("--do_lower_case", default=False, action='store_true', help="lower case of input tokens.")
    parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use MIX 16-bit float precision instead of 32-bit")
    parser.add_argument("--amp_level", default="O2", type=str, help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps 

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
 
    return args


def load_data(config):

    print("-*-"*10)
    print("current data_sign: {}".format(config.data_sign))

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

    dataset_loaders = MRCNERDataLoader(config, data_processor, label_list, tokenizer, mode="train", allow_impossible=True)
    train_dataloader = dataset_loaders.get_dataloader(data_sign="train", num_data_processor=config.num_data_processor)
    dev_dataloader = dataset_loaders.get_dataloader(data_sign="dev", num_data_processor=config.num_data_processor)
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test", num_data_processor=config.num_data_processor)
    num_train_steps = dataset_loaders.get_num_train_epochs()

    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list 



def load_model(config, num_train_steps, label_list):
    device = torch.device("cuda") 
    n_gpu = config.n_gpu
    model = BertQueryNER(config, ) 
    model.to(device)
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare optimzier 
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", 'gamma', 'beta']
    optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    # optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup_proportion,
                        t_total=num_train_steps, max_grad_norm=config.clip_grad)
    sheduler = None

    if config.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.amp_level)

    # Distributed training (should be after apex fp16 initialization)
    if config.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=True
            )

    return model, optimizer, sheduler, device, n_gpu


def train(model, optimizer, sheduler,  train_dataloader, dev_dataloader, test_dataloader, config, \
    device, n_gpu, label_list):

    dev_best_acc = 0 
    dev_best_precision = 0 
    dev_best_recall = 0 
    dev_best_f1 = 0 
    dev_best_loss = 10000000000000

    test_acc_when_dev_best = 0 
    test_pre_when_dev_best = 0 
    test_rec_when_dev_best = 0 
    test_f1_when_dev_best = 0 
    test_loss_when_dev_best = 1000000000000000

    model.train()
    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0 
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######"*10)
        print("EPOCH: ", str(idx))
        if idx != 0:
            lr_linear_decay(optimizer) 
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch) 
            input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate = batch 
            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, \
                start_positions=start_pos, end_positions=end_pos, span_positions=span_pos)

            if config.n_gpu > 1:
                loss = loss.mean()

            if config.fp16:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            model.zero_grad()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1 

            if nb_tr_steps % config.checkpoint == 0:
                print("-*-"*15)
                print("current training loss is : ")
                print(loss.item())
                model, tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config, device, n_gpu, label_list, eval_sign="dev")
                print("......"*10)
                print("DEV: loss, acc, precision, recall, f1")
                print(tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)

                if tmp_dev_f1 > dev_best_f1 :
                    dev_best_acc = tmp_dev_acc 
                    dev_best_loss = tmp_dev_loss 
                    dev_best_precision = tmp_dev_prec 
                    dev_best_recall = tmp_dev_rec 
                    dev_best_f1 = tmp_dev_f1 

                    # export model 
                    if config.export_model:
                        model_to_save = model.module if hasattr(model, "module") else model 
                        output_model_file = os.path.join(config.output_dir, "bert_finetune_model_{}_{}.bin".format(str(idx),str(nb_tr_steps)))
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("SAVED model path is :") 
                        print(output_model_file)

                    model = model.cuda().to(device)
                    model, tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model, test_dataloader, config, device, n_gpu, label_list, eval_sign="test")
                    print("......"*10)
                    print("TEST: loss, acc, precision, recall, f1")
                    print(tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1)

                    test_acc_when_dev_best = tmp_test_acc 
                    test_pre_when_dev_best = tmp_test_prec
                    test_rec_when_dev_best = tmp_test_rec
                    test_f1_when_dev_best = tmp_test_f1 
                    test_loss_when_dev_best = tmp_test_loss
                    model = model.cuda().to(device)

                print("-*-"*15)

    print("=&="*15)
    print("Best DEV : overall best loss, acc, precision, recall, f1 ")
    print(dev_best_loss, dev_best_acc, dev_best_precision, dev_best_recall, dev_best_f1)
    print("scores on TEST when Best DEV:loss, acc, precision, recall, f1 ")
    print(test_loss_when_dev_best, test_acc_when_dev_best, test_pre_when_dev_best, test_rec_when_dev_best, test_f1_when_dev_best)
    print("=&="*15)


def eval_checkpoint(model_object, eval_dataloader, config, device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader

    eval_loss = 0
    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    start_scores_lst = []
    end_scores_lst = []
    mask_lst = []
    start_gold_lst = []
    span_gold_lst = []
    end_gold_lst = []
    eval_steps = 0
    ner_cate_lst = []

    for input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
        span_pos = span_pos.to(device)

        with torch.no_grad():
            model_object.eval()
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, start_pos, end_pos, span_pos)
            start_labels, end_labels, span_scores = model_object(input_ids, segment_ids, input_mask)

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        span_pos = span_pos.to("cpu").numpy().tolist()

        start_label = start_labels.detach().cpu().numpy().tolist()
        end_label = end_labels.detach().cpu().numpy().tolist()
        span_scores = span_scores.detach().cpu().numpy().tolist()
        span_label = span_scores
        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        ner_cate_lst += ner_cate.numpy().tolist()
        eval_loss += tmp_eval_loss.mean().item()
        mask_lst += input_mask 
        eval_steps += 1

        start_pred_lst += start_label 
        end_pred_lst += end_label 
        span_pred_lst += span_label
        
        start_gold_lst += start_pos 
        end_gold_lst += end_pos 
        span_gold_lst += span_pos 

    
    if config.entity_sign == "flat":
        eval_accuracy, eval_precision, eval_recall, eval_f1 = flat_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)
    else:
        eval_accuracy, eval_precision, eval_recall, eval_f1 = nested_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)

    average_loss = round(eval_loss / eval_steps, 4)  
    eval_f1 = round(eval_f1 , 4)
    eval_precision = round(eval_precision , 4)
    eval_recall = round(eval_recall , 4) 
    eval_accuracy = round(eval_accuracy , 4)
    model_object.train()

    return model_object, average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1


def merge_config(args_config):
    model_config_path = args_config.config_path 
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, sheduler, device, n_gpu = load_model(config, num_train_steps, label_list)
    train(model, optimizer, sheduler, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)
    

if __name__ == "__main__":
    main() 
