#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: xiaoy li 
# description:
# test query_ner_evaluation 


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)



from metric.mrc_ner_evaluate import query_ner_compute_performance


def test_f1():
    pred_start = [[0, 1, 0, 0, ], [1, 0, 0, 0, 0]]
    pred_end = [[0, 0, 0, 1 ], [0, 0, 0, 0, 1]]
    pred_span = [1, 1] 

    gold_start = [[0, 0, 1, 0, ], [1, 0, 0, 0, 0]]
    gold_end =  [[0, 0, 0, 1 ], [0, 0, 0, 0, 1]]
    gold_span = [1, 1]

    ner_cate = [0, 0]
    label_lst = ["NS", "NT", "O"]
    mask_index = [[1, 1, 1, 1], [1, 1, 1, 1, 1]]

    acc, pre, rec, f1 = query_ner_compute_performance(pred_start, pred_end, pred_span, gold_start, gold_end, gold_span, ner_cate, label_lst, mask_index, dims=2)

    print("acc", acc)
    print("pre", pre)
    print("rec", rec)
    print("f1", f1)
     


def test_f2():
    # query_ner_compute_performance(pred_start, pred_end, pred_span, 
    # gold_start, gold_end, gold_span, ner_cate, label_lst, mask_idx, 
    # type_sign="flat", dims=2)
    pred_start = [[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]] 
    pred_end   = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] 
    pred_span = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ] , 
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ] ]

    gold_start = [[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]] 
    gold_end   = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] 
    gold_span = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ] , 
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ] ]

    ner_cate = [0, 0]  
    label_lst = ["NS", "NT", "O"]
    mask_index = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] 

    acc, pre, rec, f1 = query_ner_compute_performance(pred_start, pred_end, pred_span, \
        gold_start, gold_end, gold_span, ner_cate, label_lst, mask_index, type_sign="nested", dims=2)

    print("acc is : ", acc)
    print("precision is : ", pre)
    print("recall is : ", rec)
    print("f1 score is : ", f1)





if __name__ == "__main__":
    print("flat")
    test_f1()

    print("nested")
    test_f2()
