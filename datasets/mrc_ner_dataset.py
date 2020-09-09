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
        words = context.split()
        start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
        end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        # todo(yuxian): 看看是不是会有更好的截断方法
        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True)
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids
        offsets = query_context_tokens.offsets

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            # skip query tokens
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx

        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]

        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]

        match_labels = torch.zeros([self.max_length, self.max_length], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= self.max_length or end >= self.max_length:
                continue
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
    # zh datasets
    # bert_path = "/mnt/mrc/chinese_L-12_H-768_A-12"
    # json_path = "/mnt/mrc/zh_msra/mrc-ner.dev"

    # en datasets
    bert_path = "/mnt/mrc/bert-base-uncased"
    json_path = "/mnt/mrc/ace2004/mrc-ner.train"

    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file)
    dataset = MRCNERDataset(json_path=json_path, tokenizer=tokenizer)

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
