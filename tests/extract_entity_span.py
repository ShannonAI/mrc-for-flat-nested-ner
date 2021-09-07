#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: extract_entity_span.py


def get_entity_from_bmes_lst(label_list):
    """reuse the code block from
        https://github.com/jiesutd/NCRFpp/blob/105a53a321eca9c1280037c473967858e01aaa43/utils/metric.py#L73
        Many thanks to Jie Yang.
    """
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        if label_list[i] != -100:
            current_label = label_list[i].upper()
        else:
            continue
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


if __name__ == "__main__":
    label_lst = ["B-PER", "M-PER", "M-PER", "E-PER", "O", "O", "B-ORG", "M-ORG", "M-ORG", "E-ORG", "B-PER", "M-PER", "M-PER", "M-PER"]
    span_results = get_entity_from_bmes_lst(label_lst)
    print(span_results)

    label_lst = ["B-PER", "M-PER", -100, -100, "M-PER", "E-PER", -100,  "O", "O", -100, "B-ORG", -100, "M-ORG", "M-ORG", "E-ORG", "B-PER", "M-PER",
                 "M-PER", "M-PER"]
    span_results = get_entity_from_bmes_lst(label_lst)
    print(span_results)