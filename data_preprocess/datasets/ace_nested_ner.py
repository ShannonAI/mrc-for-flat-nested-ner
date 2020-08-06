#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoy li
# description:
# preprocess for ACE04/05
#
"""
{
"context": "begala dr . palmisano , again , thanks for staying with us through the break .",
"label": {
    "PER": [
        "1;2",
        "1;4",
        "11;12"],
    "ORG": [
        "1,2"]
}
}
"""

import os
import json
import sys

def reformat_annotations(input_file_path, export_file_path):
    ann_context_lst = []
    with open(input_file_path, "r") as f:
        data_lines = f.readlines()
        num_line = len(data_lines)

        for idx_pointer in range(0, num_line, 3):
            data_item_json = {}
            context = data_lines[idx_pointer]
            data_item_json["context"] = context.strip().replace("\n", "")

            entity_ann = data_lines[idx_pointer+1]
            if len(entity_ann) > 2:
                data_item_json["label"] = {}
                entity_item_line = entity_ann.split("|")
                for entity_item in entity_item_line:
                    start_end, cate = entity_item.split(" ")
                    start, end = start_end.split(",")
                    cate = cate.replace("\n", "")
                    start, end = int(start), int(end)
                    end = end - 1
                    if cate not in data_item_json["label"].keys():
                        data_item_json["label"][cate] = ["{};{}".format(str(start), str(end))]
                    else:
                        data_item_json["label"][cate].append("{};{}".format(str(start), str(end)))
            else:
                data_item_json["label"] = {}

            ann_context_lst.append(data_item_json)

    with open(export_file_path, "w") as f:
        json.dump(ann_context_lst, f, indent=2)


def main(data_sign, data_dir):
    for data_type in ["train", "dev", "test"]:
        input_data_file = os.path.join(data_dir, "{}.{}".format(data_sign, data_type))
        export_file_path = os.path.join(data_dir, "{}.{}.json".format(data_sign, data_type))
        reformat_annotations(input_data_file, export_file_path)


if __name__ == "__main__":
    data_sign = sys.argv[1]
    data_dir = sys.argv[2]
    main(data_sign, data_dir)
    # python3 ace_nested_ner.py ace2004 /home/lixiaoya/nested-ner-tacl2020-transformers/data/ace2004
    # python3 ace_nested_ner.py ace2005 /home/lixiaoya/nested-ner-tacl2020-transformers/data/ace2005