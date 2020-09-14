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
For any question, please feel free to post Github issues.<br>

## Install Requirements
`pip install -r requirements.txt`

We build our project on [pytorch-lightning.](https://github.com/PyTorchLightning/pytorch-lightning)
If you want to know more about the arguments used in our training scripts, please 
refer to [pytorch-lightning documentation.](https://pytorch-lightning.readthedocs.io/en/latest/)

## Prepare Datasets
You can [download](./ner2mrc/download.md) our preprocessed MRC-NER datasets or 
write your own preprocess scripts. We provide `ner2mrc/mrsa2mrc.py` for reference.

## Prepare Models
For English Datasets, we use [BERT-Large](https://github.com/google-research/bert)

For Chinese Datasets, we use [RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm)

## Train
The main training procedure is in `trainer.py`
Examples to start training are in `scripts/reproduce`

## Evaluate
`trainer.py` will automatically evaluate on dev set every `val_check_interval` epochs,
and save the topk checkpoints to `default_root_dir`.

To evaluate them, use `evluate.py`




## todo:
6. 调整span-match layer的层数（现在是两层？）
14. random seed
16. warmup
18. start/end也用FFN
19. 调整maxlen
20. 英文和数字全角转半角？
21. 可以在Dataset中添加sliding window的选项，直接evaluate看效果。
22. flat-ner不同种类也要去除overlap，可以先贪婪地去除一下看效果(目测没区别)
23. 英文改用BERT-Large wwm