#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bmes_decode.py

from typing import Tuple, List


class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def bmes_decode(char_label_list: List[Tuple[str, str]]) -> List[Tag]:
    """
    decode inputs to tags
    Args:
        char_label_list: list of tuple (word, bmes-tag)
    Returns:
        tags
    Examples:
        >>> x = [("Hi", "O"), ("Beijing", "S-LOC")]
        >>> bmes_decode(x)
        [{'term': 'Beijing', 'tag': 'LOC', 'begin': 1, 'end': 2}]
    """
    idx = 0
    length = len(char_label_list)
    tags = []
    while idx < length:
        term, label = char_label_list[idx]
        current_label = label[0]

        # correct labels
        if idx + 1 == length and current_label == "B":
            current_label = "S"

        # merge chars
        if current_label == "O":
            idx += 1
            continue
        if current_label == "S":
            tags.append(Tag(term, label[2:], idx, idx + 1))
            idx += 1
            continue
        if current_label == "B":
            end = idx + 1
            while end + 1 < length and char_label_list[end][1][0] == "M":
                end += 1
            if char_label_list[end][1][0] == "E":  # end with E
                entity = "".join(char_label_list[i][0] for i in range(idx, end + 1))
                tags.append(Tag(entity, label[2:], idx, end + 1))
                idx = end + 1
            else:  # end with M/B
                entity = "".join(char_label_list[i][0] for i in range(idx, end))
                tags.append(Tag(entity, label[2:], idx, end))
                idx = end
            continue
        else:
            raise Exception("Invalid Inputs")
    return tags
