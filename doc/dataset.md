## Datasets For MRC-NER 

Firstly you should transform tagging-style annoations to a set of MRC-style  `(Query, Context, Answer)` triples. Here we provide an example to show how these two steps work. We have given the queries in `python3 ./data_preprocess/dump_query2file.py` for you. Feel free to write down your own's queries.
MRC-Style datasets could be found [here](https://drive.google.com/file/d/1KHRSTL_jn5PxQqz4prQ1No2E2wWcuxOd/view?usp=sharing).

***Step 1: Query Generation***

Write down queries for entity labels in `./data_preprocess/dump_query2file.py` and run `python3 ./data_preprocess/dump_query2file.py` to dump queries to the folder `./data_preprocess/queries`. 

***Step 2: Transform tagger-style annotations to MRC-style triples*** 

Run `./data_preprocess/example/generate_data.py` to generate MRC-style data `data_preprocess/example/mrc-dev_ace05.json` and `data_preprocess/example/mrc-dev_msra.json` for ACE 2005(nested) and Chinese MSRA(flat), respectively. 


####  Nested NER 

We take ACE2005 as an example for *NESTED NER* to illustrate the process of data prepration.  

Source files for `ACE2005` contains a list of json in the format : 

```json
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
```
It assumes queries for ACE2005 should be found in `../data_preprocess/queries/en_ace05.json`. 
The path for the queries should be registered in dictionary `queries_for_dataset` of `../data_preprocess/query_map.py`. 

Run the following commands to get MRC-style data files.

```python3 
$ python3 
> from data_preprocess.generate_mrc_dataset import generate_query_ner_dataset
> source_file_path = "$PATH-TO-TAGGER-ACE05$/dev_ace05.json"
> target_file_path = "$PATH-TO-MRC-ACE05$/mrc-dev_ace05.json"
> entity_sign = "nested" #"nested" for nested-NER; "flat" for flat-NER.
> dataset_name = "en_ace2005" 
> query_sign = "default"
> generate_query_ner_dataset(source_file_path, target_file_path, entity_sign=entity_sign, dataset_name=dataset_name, query_sign=query_sign)
```

After that, `$PATH-TO-MRC-ACE05$/mrc-dev_ace05.json` contains a list of jsons: 

```json 
{
"context": "begala dr . palmisano , again , thanks for staying with us through the break .",
"end_position": [
    2,
    4,
    12
    ],
"entity_label": "PER",
"impossible": false,
"qas_id": "4.3",
"query": "3",
"span_position": [
    "1;2",
    "1;4",
    "11;12"],
"start_position": [
    1,
    1,
    11]
}
```

####  Flat NER 

Take Chinese MSRA as an example to illuatrate the process for FLAT NER. 

Source files are in CoNLL format and entities are annotated with BMES scheme : 

```
begala B-PER
dr M-PER
palmisano E-PER
, O
again O 
, O
thanks O
for O
staying O
with O
us O
through O
the O
break O
. O
```

Queries for Chinese MSRA should be found in `./data_preprocess/queries/zh_msra.json`. 
The path for the queries should be registered in dictionary `queries_for_dataset` of `./data_preprocess/query_map.py`. 

Run the following commands to get MRC-style datasets: 

```python3 
$ python3 
> from data_preprocess.generate_mrc_dataset import generate_query_ner_dataset
> source_file_path = "$PATH-TO-TAGGER-ZhMSRA$/dev_msra.bmes"
> target_file_path = "$PATH-TO-MRC-ZhMSRA$/mrc-dev_msra.json"
> entity_sign = "flat" #"nested" for nested-NER; "flat" for flat-NER.
> dataset_name = "zh_msra" 
> query_sign = "default"
> generate_query_ner_dataset(source_file_path, target_file_path, entity_sign=entity_sign, dataset_name=dataset_name, query_sign=query_sign)
```

After that, `$PATH-TO-MRC-ZhMSRA$/mrc-dev_msra.json` contains a list of jsons: 

```json 
{
"context": "begala dr . palmisano , again , thanks for staying with us through the break .",
"end_position": [2],
"entity_label": "PER",
"impossible": false,
"qas_id": "4.3",
"query": "3",
"span_position": [
    "1;2"],
"start_position": [1]
}
```
