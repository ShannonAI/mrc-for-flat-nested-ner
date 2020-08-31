#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# author: xiaoy li
# file: setup.py


from setuptools import setup, find_packages

# `pip install -e .`
NAME = "mrc_for_flat_nested_ner"
VERSION = "0.0.1"
URL = "https://github.com/ShannonAI/mrc-for-flat-nested-ner"
AUTHOR = "Xiaoya LI"
AUTHOR_EMAIL = "xiaoya_li@shannonai.com"
ZIP_SAFE = False


# 项目需要提供requirements.txt和README.md
with open('requirements.txt') as fp:
    REQUIREMENTS = fp.read().splitlines()

with open("README.md", "r")as fp:
    LONG_DESCRIPTION = fp.read()

DESCRIPTION = 'mrc-ner-for-flat-nested-ner'  # string example: description="this is a python package"
KEYWORDS = ("mrc_ner", "flat", "nested")  # string of tuple example: keywords=("test","python_package")
PLATFORMS = ["any"]  # list of string example: ["any"]


setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIREMENTS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    keywords=KEYWORDS,
    url=URL,
    packages=find_packages(exclude=["test", "log", "doc"]),
    platforms=PLATFORMS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    scripts=[],
    include_package_data=True,
    python_requires='>=3.5',
    zip_safe=ZIP_SAFE,
    # classifiers参数根据需求自行填写
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Private :: Do Not Upload",
    ),
)
