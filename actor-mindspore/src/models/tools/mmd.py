# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: mmd.py
@Date: 2023/4/27 15:25
@Author: caijianfeng
"""
import mindspore as ms


# from https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    # tiled_x = x.expand(x_size, y_size, dim)
    tiled_x = ms.numpy.tile(x, (x_size, y_size, dim))
    # tiled_y = y.expand(x_size, y_size, dim)
    tiled_y = ms.numpy.tile(y, (x_size, y_size, dim))
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return ms.ops.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd
