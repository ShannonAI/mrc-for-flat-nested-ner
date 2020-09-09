# mrc-for-flat-nested-ner

todo:
1. 拉大dropout
2. 调整max-length，保证覆盖到最长的句子；或者sliding-window
3. 测试新写的shuffle dataloader
4. 把dice loss加进来
5. GroupSampler
6. 调整span-match layer的层数（现在是两层？）
7. OHEM
8. 直接只取possible的