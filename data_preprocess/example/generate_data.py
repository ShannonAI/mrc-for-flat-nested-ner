#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# description:
# 


import os 
import sys 


ROOT_PATH = "/".join(os.path.realpath(__file__).split("/")[:-3])
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from data_preprocess.generate_mrc_dataset import generate_query_ner_dataset



def test_nested_ner():
    source_file_path = os.path.join(ROOT_PATH, "data_preprocess/test/dev_ace05.json")
    target_file_path = os.path.join(ROOT_PATH, "data_preprocess/test/mrc-dev_ace05.json")
    entity_sign = "nested"
    dataset_name = "en_ace2005"
    query_sign = "default"
    generate_query_ner_dataset(source_file_path, target_file_path, entity_sign=entity_sign, dataset_name=dataset_name, query_sign=query_sign)



def test_flat_ner():
    source_file_path = os.path.join(ROOT_PATH, "data_preprocess/test/dev_msra.bmes")
    target_file_path = os.path.join(ROOT_PATH, "data_preprocess/test/mrc-dev_msra.json")
    d_repo ="/data/nfsdata2/xiaoya/data_repo/msra_ner"
    source_file_path = os.path.join(d_repo, "bmes.test")
    target_file_path = os.path.join(d_repo, "mrc-ner.test") 
    entity_sign = "flat"
    dataset_name = "zh_msra"
    query_sign = "default"
    generate_query_ner_dataset(source_file_path, target_file_path, entity_sign=entity_sign, dataset_name=dataset_name, query_sign=query_sign)




if __name__ == "__main__":
    test_nested_ner()
    test_flat_ner() 