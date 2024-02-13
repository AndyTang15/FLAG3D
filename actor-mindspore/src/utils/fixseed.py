# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: fixseed.py
@Date: 2023/4/30 12:02
@Author: caijianfeng
"""
import numpy as np
import mindspore as ms
import random


def fixseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


SEED = 10
EVALSEED = 0
# Provoc warning: not fully functionnal yet
# torch.set_deterministic(True)
# torch.backends.cudnn.benchmark = False

fixseed(SEED)
