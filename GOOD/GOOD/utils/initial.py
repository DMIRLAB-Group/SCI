r"""Initial process for fixing all possible random seed.
"""

import random

import numpy as np
import torch

from GOOD.utils.config_reader import Union, CommonArgs, Munch
import os

def reset_random_seed(config: Union[CommonArgs, Munch]):
    r"""
    Initial process for fixing all possible random seed.

    Args:
       config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.random_seed`)


    """
    # Fix Random seed
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    os.environ['PYTHONHASHSEED'] = str(config.random_seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

    torch.cuda.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)  # 多GPU训练需要设置这个
    torch.manual_seed(config.random_seed)

    torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。

    # Default state is a training state
    torch.enable_grad()
