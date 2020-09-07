# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: mrc_ner_dataset
@time: 2020/9/6 14:27
@desc: 

"""


import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
    """
    def __init__(self, json_path, tokenizer: BertWordPieceTokenizer, max_length: int = 128):
        self.all_data = json.load(open(json_path))
        self.tokenzier = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of [query, context]
            token_type_ids: token type ids, 0 for query, 1 for context
            # start_positions: start positions of NER in tokens
            # end_positions: end positions(included) of NER in tokens
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring.
            match_labels: match labels, [seq_len, seq_len]
        """
        data = self.all_data[item]
        tokenizer = self.tokenzier

        # todo(yuxian): evaluate时可能要用到
        qas_id = data["qas_id"]
        ner_cate = data["entity_label"]

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]
        # add space offsets  todo(yuxian): 英文数据集不符合这个规则
        start_positions = [x*2 for x in start_positions]
        end_positions = [x*2 for x in end_positions]

        # todo(yuxian): 看看是不是会有更好的截断方法
        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True)
        tokens = query_context_tokens.ids[: self.max_length]
        type_ids = query_context_tokens.type_ids[: self.max_length]
        offsets = query_context_tokens.offsets[: self.max_length]
        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        new_start_positions = []
        new_end_positions = []
        label_mask = []
        for token_idx in range(len(tokens)):
            token_type = type_ids[token_idx]
            # skip query tokens
            if token_type == 0:
                label_mask.append(0)
                continue
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                label_mask.append(0)
                continue
            if token_start in start_positions:
                new_start_positions.append(token_idx)
            if token_end - 1 in end_positions:
                new_end_positions.append(token_idx)
            label_mask.append(1)

        assert (len(new_start_positions) == len(new_end_positions) == len(start_positions)
                or len(query_context_tokens) > self.max_length)
        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]
        match_labels = torch.zeros([self.max_length, self.max_length], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            match_labels[start, end] = 1

        return [
            torch.LongTensor(self.pad(tokens, 0)),
            torch.LongTensor(self.pad(type_ids, 1)),
            torch.LongTensor(self.pad(start_labels)),
            torch.LongTensor(self.pad(end_labels)),
            torch.LongTensor(self.pad(label_mask)),
            match_labels
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def run_dataset():
    """test dataset"""
    import os
    json_path = "/data/yuxian/mrc/zh_msra/mrc-ner.dev"
    bert_path = "/data/yuxian/mrc/chinese_L-12_H-768_A-12"
    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file)
    dataset = MRCNERDataset(json_path=json_path, tokenizer=tokenizer)
    # for d in dataset:
        # tokens = d["tokens"].tolist()
        # start_positions = d["start_positions"].tolist()
        # end_positions = d["end_positions"].tolist()
    for tokens, token_type_ids, start_labels, end_labels, label_mask, match_labels in dataset:
        tokens = tokens.tolist()
        start_positions, end_positions = torch.where(match_labels > 0)
        start_positions = start_positions.tolist()
        end_positions = end_positions.tolist()
        if not start_positions:
            continue
        print("="*20)
        print(tokenizer.decode(tokens, skip_special_tokens=False))
        for start, end in zip(start_positions, end_positions):
            print(tokenizer.decode(tokens[start: end+1]))


if __name__ == '__main__':
    run_dataset()
