#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Description:
# bert_tokenizer to solve the span of tagging 


from pytorch_pretrained_bert.tokenization import BertTokenizer 



def whitespace_tokenize(text):
    """
    Desc:
        runs basic whitespace cleaning and splitting on a piece of text
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens 



class BertTokenizer4Tagger(BertTokenizer):
    """
    Desc:
        slove the problem of tagging span can not fit after run word_piece tokenizing 
    """
    def __init__(self, vocab_file, do_lower_case=False, max_len=None, 
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):

        super(BertTokenizer4Tagger, self).__init__(vocab_file, do_lower_case=do_lower_case, 
            max_len=max_len, never_split=never_split) 


    def tokenize(self, text, label_lst=None):
        """
        Desc:
            text: 
            label_lst: ["B", "M", "E", "S", "O"]
        """

        split_tokens = []
        split_labels = []

        if label_lst is None:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
            return split_tokens 


        for token, label in zip(self.basic_tokenizer.tokenize(text), label_lst):
            # cureent token should be 1 single word 
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            if len(sub_tokens) > 1:
                for tmp_idx, tmp_sub_token in enumerate(sub_tokens):
                    if tmp_idx == 0:
                        split_tokens.append(tmp_sub_token)
                        split_labels.append(label)
                    else:
                        split_tokens.append(tmp_sub_token)
                        split_labels.append("X")
            else:
                split_tokens.append(sub_token)
                split_labels.append(label)

        return split_tokens, split_labels 
