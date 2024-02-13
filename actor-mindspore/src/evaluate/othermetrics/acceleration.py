# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: acceleration.py
@Date: 2023/4/27 20:43
@Author: caijianfeng
"""
import mindspore as ms
import mindspore.ops as ops
import numpy as np

from src.utils.tensors import lengths_to_mask


def calculate_acceletation(motionloader, device, xyz):
    # for now even if it is not xyz, the acceleration is one the euclidian/pose
    outfeat = "output_xyz" if xyz else "output"

    sum_acc = 0
    num_acc = 0
    for batch in motionloader:
        motion = batch[outfeat].permute(0, 3, 1, 2)
        bs, num_frames, njoints, nfeats = motion.shape

        velocity = motion[:, 1:] - motion[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        # acceleration_normed = torch.linalg.norm(acceleration, axis=3)
        # TODO: 寻找与 torch.linalg.norm 对应的 mindspore API
        acceleration_normed = ops.norm(acceleration, axis=3)
        lengths = batch["lengths"]
        mask = lengths_to_mask(lengths - 2)  # because acceleration

        usefull_accs_n = acceleration_normed[mask]
        sum_acc += usefull_accs_n.sum().item()
        num_acc += np.prod(usefull_accs_n.shape)

    acceleration = sum_acc / num_acc
    return acceleration
