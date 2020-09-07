# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: TruncateDatset
@time: 2020/7/3 17:48

    Truncate Datset, mostly used for debug
"""

from torch.utils.data import Dataset


class TruncateDataset(Dataset):
    """Truncate dataset to certain num"""
    def __init__(self, dataset: Dataset, max_num: int = 100):
        self.dataset = dataset
        self.max_num = min(max_num, len(self.dataset))

    def __len__(self):
        return self.max_num

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, item):
        """other dataset func"""
        return getattr(self.dataset, item)
