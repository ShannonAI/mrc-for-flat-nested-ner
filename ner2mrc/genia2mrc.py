# encoding: utf-8


import os
from utils.bmes_decode import bmes_decode
import json


def convert_file(input_file, output_file, tag2query_file):
    """
    Convert GENIA(xiaoya) data to MRC format
    """
    all_data = json.load(open(input_file))
    tag2query = json.load(open(tag2query_file))

    output = []
    origin_count = 0
    new_count = 0

    for data in all_data:
        origin_count += 1
        context = data["context"]
        label2positions = data["label"]
        for tag_idx, (tag, query) in enumerate(tag2query.items()):
            positions = label2positions.get(tag, [])
            mrc_sample = {
                "context": context,
                "query": query,
                "start_position": [int(x.split(";")[0]) for x in positions],
                "end_position": [int(x.split(";")[1]) for x in positions],
                "qas_id": f"{origin_count}.{tag_idx}"
            }
            output.append(mrc_sample)
            new_count += 1

    json.dump(output, open(output_file, "w"), ensure_ascii=False, indent=2)
    print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")


def main():
    genia_raw_dir = "/mnt/mrc/genia/genia_raw"
    genia_mrc_dir = "/mnt/mrc/genia/genia_raw/mrc_format"
    tag2query_file = "queries/genia.json"
    os.makedirs(genia_mrc_dir, exist_ok=True)
    for phase in ["train", "dev", "test"]:
        old_file = os.path.join(genia_raw_dir, f"{phase}.genia.json")
        new_file = os.path.join(genia_mrc_dir, f"mrc-ner.{phase}")
        convert_file(old_file, new_file, tag2query_file)


if __name__ == '__main__':
    main()
