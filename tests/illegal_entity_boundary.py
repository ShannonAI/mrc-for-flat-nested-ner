#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: illegal_entity_boundary.py

from transformers import AutoTokenizer
from tokenizers import BertWordPieceTokenizer

def load_dataexamples(file_path, ):
    with open(file_path, "r") as f:
        datalines = f.readlines()

    sentence_collections = []
    sentence_label_collections = []
    word_collections = []
    word_label_collections = []

    for data_item in datalines:
        data_item = data_item.strip()
        if len(data_item) != 0:
            word, label = tuple(data_item.split(" "))
            word_collections.append(word)
            word_label_collections.append(label)
        else:
            sentence_collections.append(word_collections)
            sentence_label_collections.append(word_label_collections)
            word_collections = []
            word_label_collections = []

    return sentence_collections, sentence_label_collections


def find_data_instance(file_path, search_string):
    sentence_collections, sentence_label_collections = load_dataexamples(file_path)

    for sentence_lst, label_lst in zip(sentence_collections, sentence_label_collections):
        sentence_str = "".join(sentence_lst)
        if search_string in sentence_str:
            print(sentence_str)
            print("-"*10)
            print(sentence_lst)
            print(label_lst)
            print("=*"*10)

def find_illegal_entity(model_path, is_chinese=True, do_lower_case=True):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, do_lower_case=do_lower_case)
    query = "组织机构"
    context_tokens = ['１', '美', '亚', '股', '份', '３', '２', '．', '６', '６', '２', '民', '族', '集', '团', '２', '２', '．', '３',
                      '８', '３', '鲁', '石', '化', 'Ａ', '１', '９', '．', '１', '１', '４', '四', '川', '湖', '山', '１', '７', '．',
                      '０', '９', '５', '太', '原', '刚', '玉', '１', '０', '．', '５', '８', '１', '咸', '阳', '偏', '转', '１', '６',
                      '．', '１', '１', '２', '深', '华', '发', 'Ａ', '１', '５', '．', '６', '６', '３', '渝', '开', '发', 'Ａ', '１',
                      '５', '．', '５', '２', '４', '深', '发', '展', 'Ａ', '１', '３', '．', '８', '９', '５', '深', '纺', '织', 'Ａ',
                      '１', '３', '．', '２', '２', '１', '太', '极', '实', '业', '２', '３', '．', '２', '２', '２', '友', '好', '集',
                      '团', '２', '２', '．', '１', '４', '３', '双', '虎', '涂', '料', '２', '０', '．', '２', '０', '４', '新', '潮',
                      '实', '业', '１', '５', '．', '５', '８', '５', '信', '联', '股', '份', '１', '２', '．', '５', '７', '１', '氯',
                      '碱', '化', '工', '２', '１', '．', '１', '７', '２', '百', '隆', '股', '份', '１', '５', '．', '６', '４', '３',
                      '贵', '华', '旅', '业', '１', '５', '．', '１', '５', '４', '南', '洋', '实', '业', '１', '４', '．', '５', '０',
                      '５', '福', '建', '福', '联', '１', '３', '．', '８', '０']
    # context = "１美亚股份３２．６６２民族集团２２．３８３鲁石化Ａ１９．１１４四川湖山１７．０９５太原刚玉１０．５８１咸阳偏转１６．１１２深华发Ａ１５．６６３渝开发Ａ１５．５２４深发展Ａ１３．８９５深纺织Ａ１３．２２１太极实业２３．２２２友好集团２２．１４３双虎涂料２０．２０４新潮实业１５．５８５信联股份１２．５７１氯碱化工２１．１７２百隆股份１５．６４３贵华旅业１５．１５４南洋实业１４．５０５福建福联１３．８０"
    context = "".join(context_tokens)

    labels = ['O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O',
              'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT',
              'M-NT', 'E-NT', 'O', \
              'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT',
              'M-NT', \
              'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O',
              'B-NT', 'M-NT', \
              'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O',
              'O', 'B-NT', \
              'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O',
              'O', 'O', \
              'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O',
              'O', 'O', \
              'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT',
              'O', 'O', \
              'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT',
              'E-NT', \
              'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT',
              'M-NT', 'M-NT', \
              'E-NT', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NT', 'M-NT', 'M-NT', 'E-NT', 'O', 'O', 'O', 'O', 'O']

    start_positions = []
    end_positions = []
    origin_tokens = context_tokens
    print("check labels in ")
    print(len(origin_tokens))
    print(len(labels))

    for label_idx, label_item in enumerate(labels):
        if "B-" in label_item:
            start_positions.append(label_idx)
        if "S-" in label_item:
            end_positions.append(label_idx)
            start_positions.append(label_idx)
        if "E-" in label_item:
            end_positions.append(label_idx)

    print("origin entity tokens")
    for start_item, end_item in zip(start_positions, end_positions):
        print(origin_tokens[start_item: end_item + 1])

    query_context_tokens = tokenizer.encode_plus(query, context,
                                                 add_special_tokens=True,
                                                 max_length=500,
                                                 return_overflowing_tokens=True,
                                                 return_token_type_ids=True)

    if tokenizer.pad_token_id in query_context_tokens["input_ids"]:
        non_padded_ids = query_context_tokens["input_ids"][
                         : query_context_tokens["input_ids"].index(tokenizer.pad_token_id)]
    else:
        non_padded_ids = query_context_tokens["input_ids"]

    non_pad_tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
    first_sep_token = non_pad_tokens.index("[SEP]")
    end_sep_token = len(non_pad_tokens) - 1
    new_start_positions = []
    new_end_positions = []
    if len(start_positions) != 0:
        for start_index, end_index in zip(start_positions, end_positions):
            if is_chinese:
                answer_text_span = " ".join(context[start_index: end_index + 1])
            else:
                answer_text_span = " ".join(context.split(" ")[start_index: end_index + 1])
            new_start, new_end = _improve_answer_span(query_context_tokens["input_ids"], first_sep_token, end_sep_token,
                                                      tokenizer, answer_text_span)
            new_start_positions.append(new_start)
            new_end_positions.append(new_end)
    else:
        new_start_positions = start_positions
        new_end_positions = end_positions

    # clip out-of-boundary entity positions.
    new_start_positions = [start_pos for start_pos in new_start_positions if start_pos < 500]
    new_end_positions = [end_pos for end_pos in new_end_positions if end_pos < 500]

    print("print tokens :")
    for start_item, end_item in zip(new_start_positions, new_end_positions):
        print(tokenizer.convert_ids_to_tokens(query_context_tokens["input_ids"][start_item: end_item + 1]))


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text, return_subtoken_start=False):
    """Returns tokenized answer spans that better match the annotated answer."""
    doc_tokens = [str(tmp) for tmp in doc_tokens]
    answer_tokens = tokenizer.encode(orig_answer_text, add_special_tokens=False)
    tok_answer_text = " ".join([str(tmp) for tmp in answer_tokens])
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end+1)])
            if text_span == tok_answer_text:
                if not return_subtoken_start:
                    return (new_start, new_end)
                tokens = tokenizer.convert_ids_to_tokens(doc_tokens[new_start: (new_end + 1)])
                if "##" not in tokens[-1]:
                    return (new_start, new_end)
                else:
                    for idx in range(len(tokens)-1, -1, -1):
                        if "##" not in tokens[idx]:
                            new_end = new_end - (len(tokens)-1 - idx)
                            return (new_start, new_end)

    return (input_start, input_end)


if __name__ == "__main__":
    file_path = "/data/xiaoya/datasets/ner/msra/train.char.bmes"
    search_string = "美亚股份"
    find_data_instance(file_path, search_string)

    print("=%"*20)
    print("check entity boundary")
    print("=&"*20)

    model_path = "/data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12"
    find_illegal_entity(model_path)
