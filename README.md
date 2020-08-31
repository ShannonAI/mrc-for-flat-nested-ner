# A Unified MRC Framework for Named Entity Recognition 
The repository contains the code of the recent research advances in [Shannon.AI](http://www.shannonai.com). 

**A Unified MRC Framework for Named Entity Recognition** <br>
Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu and Jiwei Li<br>
In ACL 2020. [paper](https://arxiv.org/abs/1910.11476)<br>
If you find this repo helpful, please cite the following:
```latex
@article{li2019unified,
  title={A Unified MRC Framework for Named Entity Recognition},
  author={Li, Xiaoya and Feng, Jingrong and Meng, Yuxian and Han, Qinghong and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1910.11476},
  year={2019}
}
```
For any question, please feel free to contact xiaoya_li@shannonai.com or post Github issues.<br>


## Content
1. [Overview](#overview)
2. [Experimental Results on Flat/Nested NER Datasets](#experimental-results-on-flat-nested-ner-datasets)
3. [Requirements](#requirements)
4. [Data Preprocess](#data-preprocess)
5. [Training BERT MRC-NER Model](#training-bert-mrc-ner-model)
6. [Evaluating the Trained Model](#evaluating-the-trained-model)
7. [Descriptions of Directories](#descriptions-of-directories)
8. [Contact](#contact)


## Overview 

The task of NER is normally divided into **nested** NER and **flat** NER depending on whether named entities are nested or not. Instead of treating the task of NER as a sequence labeling problem, we propose to formulate it as a SQuAD-style machine reading comprehension (MRC) task. <br>

For example, the task of assigning the [PER] label to *"[Washington] was born into slavery on the farm of James Burroughs"* is formalized as answering the question *"Which person is mentioned in the text?"*. <br>

By unifying flat and nested NER under an MRC framework, we're able to gain a huge improvement on both flat and nested NER datasets, which achives SOTA results.

We use `MRC-NER` to denote the proposed framework. <br>
Here are some of the **highlights**:

1. *MRC-NER* works better than BERT-Tagger with less training data. 
2. *MRC-NER* is capable of handling both flat and nested NER tasks under a unified framework.  
3. *MRC-NER* has a better zero-shot learning ability which can predicts labels unseen from the training set.  
4. The query encodes prior information about the entity category to extract and has the potential to disambiguate similar classes. 

## Experimental Results on Flat/Nested NER Datasets
Experiments are conducted on both *Flat* and *Nested* NER datasets. The proposed method achieves vast amount of performance boost over current SOTA models. <br>
We only list comparisons between our proposed method and previous SOTA in terms of span-level micro-averaged F1-score here. 
For more comparisons and span-level micro Precision/Recall scores, please check out our [paper](https://arxiv.org/abs/1910.11476.pdf). <br> 
### Flat NER Datasets
Evaluations are conducted on the widely-used bechmarks: `CoNLL2003`, `OntoNotes 5.0` for English; `MSRA`, `OntoNotes 4.0` for Chinese. We achieve new SOTA results on `OntoNotes 5.0`, `MSRA` and  `OntoNotes 4.0`, and comparable results on `CoNLL2003`.

| Dataset |  Eng-OntoNotes5.0 | Zh-MSRA | Zh-OntoNotes4.0 | 
|---|---|---|---|
| Previous SOTA | 89.16 | 95.54  | 81.63 | 
| Our method |  **91.11** | **95.75** | **82.11** | 
|  |  **(+1.95)** | **(+0.21)** | **(+0.48)** | 

### Nested NER Datasets
Evaluations are conducted on the widely-used `ACE 2004`, `ACE 2005`, `GENIA`, `KBP-2017` English datasets.

| Dataset | ACE 2004 | ACE 2005 | GENIA | KBP-2017 | 
|---|---|---|---|---|
| Previous SOTA | 84.7 | 84.33 | 78.31  | 74.60 | 
| Our method | **85.98** | **86.88** | **83.75** | **80.97** | 
|  | **(+1.28)** | **(+2.55)** | **(+5.44)** | **(+6.37)** | 

Previous SOTA:
 
* [DYGIE](https://www.aclweb.org/anthology/N19-1308/) for ACE 2004.
* [Seq2Seq-BERT](https://www.aclweb.org/anthology/P19-1527/) for ACE 2005 and GENIA.
* [ARN](https://www.aclweb.org/anthology/P19-1511/) for KBP2017. 

## Requirements

- GPU machines with Python 3.6.  <br> 
	- Install [PyTorch](https://pytorch.org/) >= 1.1.0
	- Run `pip3 install -r requirements.txt`

-  Download the pretrained BERT checkpoints and transform the checkpoints to PyTorch. <br>
	- Run the following command and download the pretrained BERT checkpoints. 
	`bash ./script/data/download_pretrained_model.sh <dir_to_save_pretrained_ckpt> <model_name>` <br> 
	`<model_name>` should take the value of `[en_bert_cl, en_bert_wwm_cl, zh_bert]`.

	- Run the following command and transform the checkpoints from tensorflow (.ckpt) to pytorch (.bin). <br>

**NOTICE**: need to install `tensorflow-gpu==1.15`

`bash ./script/data/convert_checkpoints_from_tf_to_pytorch.sh <model_sign> <dir_to_bert_model>` <br> 

`<model_sign>` should take the value of `[zh_bert, en_bert_cased_large, en_bert_wwm_cased_large]`. 

- For faster training, install NVIDIA's [Apex](https://github.com/NVIDIA/apex) library:
	
```bash
git clone https://github.com/NVIDIA/apex
cd apex 
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
``` 
**NOTICE**: The program stops running and raises a `StopIteration ERROR` exception if you want to train the model on Multi-gpu with `torch==1.5.x`.
The solution is to increase the values of `--gradient_accumulation_steps` and `--train_batch_size`. 
Please refer the [link](https://github.com/amdegroot/ssd.pytorch/issues/214) for more details.

## Data Preprocess 

You can [download](./doc/download.md) our preprocessed MRC-NER datasets or follow the [instruction](./doc/dataset.md) to build your own datasets. <br> 
For **large** datasets (English OntoNotes 5.0), run the following command and generate cached datasets before experiments. <br> 
```bash 
./script/data/transform_mrc_datasets_to_cache.sh <data_sign> <max_input_length> <dir_to_mrc_ner_datasets> <num_data_processor> <dir_to_bert_model>
```

* `<data_sign>` should take the value of `[conll03, zh_msra, zh_onto, en_onto, genia, ace2004, ace2005, resume]`.

* `<num_data_processor>` denotes the number of processor for transforming data examples to features. 

## Training BERT MRC-NER Model

You can directly use the following commands to train the **MRC-NER** model with some minor changes.<br>
`data_sign` should take the value of  `[conll03, zh_msra, zh_onto, en_onto, genia, ace2004, ace2005, kbp17]`. <br> 
`entity_sign` should take the value of `[flat, nested]`. <br> 

```bash 

REPO_PATH=/PATH-TO-REPO/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
CONFIG_PATH=${FOLDER_PATH}/config/zh_bert.json
DATA_PATH=/PATH-TO-BERT_MRC-DATA/zh_ontonotes4
BERT_PATH=/PATH-TO-BERT-CHECKPOINTS/chinese_L-12_H-768_A-12
OUTPUT_PATH=/PATH-TO-SAVE-MODEL-CKPT

CUDA_VISIBLE_DEVICES=0 nohup python3 $REPO_PATH/run/train_bert_mrc.py \
--optimizer_type adamw \
--lr_scheduler_type ladder \
--lr_min 8e-6 \
--loss_type ce \
--weight_decay 0.01 \
--max_grad_norm 1.0 \
--data_dir $DATA_PATH \
--n_gpu 1 \
--entity_sign flat \
--num_data_processor 1 \
--data_sign zh_onto \
--logfile_name log.txt \
--bert_model $BERT_PATH \
--config_path $CONFIG_PATH \
--output_dir $OUTPUT_PATH \
--dropout 0.1 \
--checkpoint 600 \
--max_seq_length 256 \
--train_batch_size 12 \
--dev_batch_size 12 \
--test_batch_size 12 \
--learning_rate 1e-5 \
--weight_start 1.0 \
--weight_end 1.0 \
--weight_span 1.0 \
--num_train_epochs 8 \
--seed 2333 \
--warmup_steps 500 \
--gradient_accumulation_steps 2 \
--only_eval_dev &
```

** Notice: ** We recommend set `--num_data_processor $NUM_DATA_PROCESSOR` to `1` for small datasets and enlarge for large datasets like English OntoNotes 5.0. 

## Evaluating the Trained Model

You can directly use the following commands to evaluate the **MRC-NER** model after training.<br>
`data_sign`should take the value of `[conll03, zh_msra, zh_onto, en_onto, genia, ace2004, ace2005, kbp17, resume]`. <br> 
`entity_sign` should take the value of `[flat, nested]`. <br> 

```bash 

REPO_PATH=/PATH-TO-REPO/mrc-for-flat-nested-ner 
CONFIG_PATH=${FOLDER_PATH}/config/zh_bert.json
DATA_PATH=/PATH-TO-BERT_MRC-DATA/zh_ontonotes4
BERT_PATH=/PATH-TO-BERT-CHECKPOINTS/chinese_L-12_H-768_A-12
EVAL_MODEL_PATH=/PATH-TO-SAVED-MODEL-CKPT/bert_finetune_model.bin

CUDA_VISIBLE_DEVICES=0  python3 $REPO_PATH/run/evaluate_mrc_ner.py \
--config_path $CONFIG_PATH \
--data_dir $DATA_PATH \
--bert_model $BERT_PATH \
--saved_model $EVAL_MODEL_PATH \
--logfile_path eval_log.txt \
--data_sign zh_onto \
--entity_sign flat \
--num_data_processor 1 \
--max_seq_length 260 \
--test_batch_size 32 \
--dropout 0.1 \
--weight_start 1 \
--weight_end 1 \
--weight_span 1 \
--entity_threshold 0.15 \
--seed 2333 \
--n_gpu 1
```

## Descriptions of Directories 

Name | Descriptions 
----------- | ------------- 
log | A collection of training logs in experments.   
script |  Shell files help to reproduce our results.  
data_preprocess | Files to generate MRC-NER train/dev/test datasets. 
metric | Evaluation metrics for Flat/Nested NER. 
model | An implementation of MRC-NER based on Pytorch.
layer | Components for building MRC-NER model. 
data_loader | Funcs for loading MRC-style datasets.  
run | Train / Evaluate MRC-NER models.
config | Config files for BERT models. 


## Contact 

Feel free to discuss papers/code with us through issues/emails!
xiaoya_li AT shannonai.com









