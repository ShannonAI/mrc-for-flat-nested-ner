# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: radom_seed
@time: 2020/7/9 15:53

    这一行开始写关于本文件的说明与解释
"""

import numpy as np
import torch


def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # without this line, x would be different in every execution.
    set_random_seed(0)

    x = np.random.random()
    print(x)
