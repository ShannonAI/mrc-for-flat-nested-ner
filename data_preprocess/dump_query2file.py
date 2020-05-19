#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# description:
# 


import os 
import json 


msra = {
  "default": {
    "NR": "人名和虚构的人物形象",
    "NS": "按照地理位置划分的国家,城市,乡镇,大洲",
    "NT": "组织包括公司,政府党派,学校,政府,新闻机构"
  },
  "labels": [
    "NS",
    "NR",
    "NT"
  ]
}


ace2005 = {
  "default": {
    "FAC": "facility entities are limited to buildings and other permanent man-made structures such as buildings, airports, highways, bridges.",
    "GPE": "geographical political entities are geographical regions defined by political and or social groups such as countries, nations, regions, cities, states, government and its people. ",
    "LOC": "location entities are limited to geographical entities such as geographical areas and landmasses, mountains, bodies of water, and geological formations.",
    "ORG": "organization entities are limited to companies, corporations, agencies, institutions and other groups of people.",
    "PER": "a person entity is limited to human including a single individual or a group.",
    "VEH": "vehicle entities are physical devices primarily designed to move, carry, pull or push the transported object such as helicopters, trains, ship and motorcycles.",
    "WEA": "weapon entities are limited to physical devices such as instruments for physically harming such as guns, arms and gunpowder."
  },
  "labels": [
    "GPE",
    "ORG",
    "PER",
    "FAC",
    "VEH",
    "LOC",
    "WEA"
  ]
}

ace04 = {
  "default": {
    "FAC": "facility entities are limited to buildings and other permanent man-made structures such as buildings, airports, highways, bridges.",
    "GPE": "geographical political entities are geographical regions defined by political and or social groups such as countries, nations, regions, cities, states, government and its people. ",
    "LOC": "location entities are limited to geographical entities such as geographical areas and landmasses, mountains, bodies of water, and geological formations.",
    "ORG": "organization entities are limited to companies, corporations, agencies, institutions and other groups of people.",
    "PER": "a person entity is limited to human including a single individual or a group.",
    "VEH": "vehicle entities are physical devices primarily designed to move, carry, pull or push the transported object such as helicopters, trains, ship and motorcycles.",
    "WEA": "weapon entities are limited to physical devices such as instruments for physically harming such as guns, arms and gunpowder."
  },
  "labels": [
    "GPE",
    "ORG",
    "PER",
    "FAC",
    "VEH",
    "LOC",
    "WEA"
  ]
}


zh_ontonotes4 = {
  "default": {
    "GPE": "按照国家,城市,州县划分的地理区域",
    "LOC": "山脉,河流自然景观的地点",
    "ORG": "组织包括公司,政府党派,学校,政府,新闻机构",
    "PER": "人名和虚构的人物形象"
  },
  "labels": [
    "LOC",
    "PER",
    "GPE",
    "ORG"
  ]
}


if __name__ == "__main__":

    repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])

    with open(os.path.join(repo_path, "data_preprocess/queries/zh_msra.json"), "w") as f:
        json.dump(msra, f, sort_keys=True, indent=2, ensure_ascii=False)

    with open(os.path.join(repo_path, "data_preprocess/queries/zh_ontonotes4.json"), "w") as f:
        json.dump(zh_ontonotes4, f, sort_keys=True, indent=2, ensure_ascii=False)


