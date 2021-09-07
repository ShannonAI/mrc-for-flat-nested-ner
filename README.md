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
For any question, please feel free to post Github issues. <br>

## Install Requirements

* The code requires Python 3.6+.

* If you are working on a GPU machine with CUDA 10.1, please run `pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html` to install PyTorch. If not, please see the [PyTorch Official Website](https://pytorch.org/) for instructions.

* Then run the following script to install the remaining dependenices: `pip install -r requirements.txt`

We build our project on [pytorch-lightning.](https://github.com/PyTorchLightning/pytorch-lightning)
If you want to know more about the arguments used in our training scripts, please 
refer to [pytorch-lightning documentation.](https://pytorch-lightning.readthedocs.io/en/latest/)

### Baseline: BERT-Tagger 

We release code, [scripts](./scripts/bert_tagger/reproduce) and [datafiles](./ner2mrc/download.md) for fine-tuning BERT and treating NER as a sequence labeling task. <br>

### MRC-NER: Prepare Datasets

You can [download](./ner2mrc/download.md) the preprocessed MRC-NER datasets used in our paper. <br>
For custom datasets, please use `ner2mrc/mrsa2mrc.py` to transform your BMES NER annotations to MRC-format. 

### MRC-NER: Training

The main training procedure is in `train/mrc_ner_trainer.py`

Scripts for reproducing our experimental results can be found in the `./scripts/mrc_ner/reproduce/` folder. 
Note that you need to change `DATA_DIR`, `BERT_DIR`, `OUTPUT_DIR` to your own dataset path, bert model path and log path, respectively.  <br> 
For example, run `./scripts/mrc_ner/reproduce/ace04.sh` will start training MRC-NER models and save intermediate log to `$OUTPUT_DIR/train_log.txt`. <br> 
During training, the model trainer will automatically evaluate on the dev set every `val_check_interval` epochs,
and save the topk checkpoints to `$OUTPUT_DIR`. <br> 

### MRC-NER: Evaluation

After training, you can find the best checkpoint on the dev set according to the evaluation results in `$OUTPUT_DIR/train_log.txt`. <br> 
Then run `python3 evaluate/mrc_ner_evaluate.py $OUTPUT_DIR/<best_ckpt_on_dev>.ckpt  $OUTPUT_DIR/lightning_logs/<version_0/hparams.yaml>` to evaluate on the test set with the best checkpoint chosen on dev. 

### MRC-NER: Inference 

Code for inference using the trained MRC-NER model can be found in `inference/mrc_ner_inference.py` file. <br>
For flat NER, we provide the inference script in [flat_inference.sh](./scripts/mrc_ner/flat_inference.sh) <br>
For nested NER, we provide the inference script in [nested_inference.sh](./scripts/mrc_ner/nested_inference.sh) 