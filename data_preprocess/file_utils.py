#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI  
# Description;
# some usefule file utils 


def export_conll(sentence, label, export_file_path, dim=2):
    """
    Args:
        sentence: a list of sentece of chars [["北", "京", "天", "安", "门"], ["真", "相", "警", 告"]]
        label: a list of labels [["B", "M", "E", "S", "O"], ["O", "O", "S", "S"]] 
    Desc:
        export tagging data into conll format 
    """
    with open(export_file_path, "w") as f:
        for idx, (sent_item, label_item) in enumerate(zip(sentence, label)): 
            for char_idx, (tmp_char, tmp_label) in enumerate(zip(sent_item, label_item)):
                f.write("{} {}\n".format(tmp_char, tmp_label))
            f.write("\n")


def load_conll(data_path):
    """
    Desc:
        load data in conll format 
    Returns:
        [([word1, word2, word3, word4], [label1, label2, label3, label4]), 
        ([word5, word6, word7, wordd8], [label5, label6, label7, label8])]
    """
    dataset = []
    with open(data_path, "r") as f:
        words, tags = [], []
        # for each line of the file correspond to one word and tag 
        for line in f:
            if line != "\n":
                # line = line.strip()
                word, tag = line.split(" ")
                word = word.strip()
                tag = tag.strip()
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print("an exception was raise! skipping a word")
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []

    return dataset 


def dump_tsv(data_lines, data_path):
    """
    Desc:
        dump data into tsv format for TAGGING data
    Input:
        the format of data_lines is:
            [([word1, word2, word3, word4], [label1, label2, label3, label4]), 
            ([word5, word6, word7, word8, word9], [label5, label6, label7, label8, label9]), 
            ([word10, word11, word12, ], [label10, label11, label12])]
    """
    print("dump dataliens into TSV format : ")
    with open(data_path, "w") as f:
        for data_item in data_lines:
            data_word, data_tag = data_item 
            data_str = " ".join(data_word)
            data_tag = " ".join(data_tag)
            f.write(data_str + "\t" + data_tag + "\n")
        print("dump data set into data path")
        print(data_path)




