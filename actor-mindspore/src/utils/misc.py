# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: misc.py
@Date: 2023/4/28 11:32
@Author: caijianfeng
"""
import mindspore as ms
# from mindspore import ops

def is_tensor(obj):
    return isinstance(obj, ms.Tensor)

def to_numpy(tensor):
    if is_tensor(tensor):
        return tensor.numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_ms(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return ms.Tensor.from_numpy(ndarray).astype(ms.float32)
    elif not is_tensor(ndarray):
        raise ValueError("Cannot convert {} to mindspore tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

