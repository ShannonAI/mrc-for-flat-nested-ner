# encoding: utf-8


import os
from utils.bmes_decode import bmes_decode
import json


def convert_file(input_file, output_file, tag2query_file):
    """
    Convert MSRA raw data to MRC format
    """
    origin_count = 0
    new_count = 0
    tag2query = json.load(open(tag2query_file))
    mrc_samples = []
    with open(input_file) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            origin_count += 1
            src, labels = line.split("\t")
            tags = bmes_decode(char_label_list=[(char, label) for char, label in zip(src.split(), labels.split())])
            for label, query in tag2query.items():
                mrc_samples.append(
                    {
                        "context": src,
                        "start_position": [tag.begin for tag in tags if tag.tag == label],
                        "end_position": [tag.end-1 for tag in tags if tag.tag == label],
                        "query": query
                    }
                )
                new_count += 1

    json.dump(mrc_samples, open(output_file, "w"), ensure_ascii=False, sort_keys=True, indent=2)
    print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")


def main():
    msra_raw_dir = "/mnt/mrc/zh_msra_yuxian"
    msra_mrc_dir = "/mnt/mrc/zh_msra_yuxian/mrc_format"
    tag2query_file = "queries/zh_msra.json"
    os.makedirs(msra_mrc_dir, exist_ok=True)
    for phase in ["train", "dev", "test"]:
        old_file = os.path.join(msra_raw_dir, f"{phase}.tsv")
        new_file = os.path.join(msra_mrc_dir, f"mrc-ner.{phase}")
        convert_file(old_file, new_file, tag2query_file)


if __name__ == '__main__':
    main()
