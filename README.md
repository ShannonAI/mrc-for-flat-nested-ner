# mrc-for-flat-nested-ner

todo:
1. 拉大dropout
2. 调整max-length，保证覆盖到最长的句子；或者sliding-window
4. 把dice loss加进来
6. 调整span-match layer的层数（现在是两层？）
9. activation 换用gelu
10. ce->bce
11. 验证weight_sum没有问题(lr相应扩大)
12. optimizer, 让SGD震荡起来=-=
13. 尝试只用start/end的golden/pred train match，注意要调整lr
14. random seed
15. 中文数据集的tokenize要与BERT预训练一致，现在应该有数字被单独分开。

## Convert format
`./ner2mrc/`

## Train
`trainer.py`

examples: `./scripts/`

## Evaluate
`evluate.py`
