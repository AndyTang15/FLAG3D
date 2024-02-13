# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: diversity.py
@Date: 2023/4/27 20:30
@Author: caijianfeng
"""
import mindspore as ms
import mindspore.ops as ops
import numpy as np


# from action2motion
def calculate_diversity_multimodality(activations, labels, num_labels):
    # print(type(activations), ';', activations.shape)
    diversity_times = 200
    multimodality_times = 20
    labels = labels.long()
    num_motions = len(labels)

    diversity = 0

    first_indices = ms.Tensor.from_numpy(np.random.randint(0, num_motions, diversity_times))
    second_indices = ms.Tensor.from_numpy(np.random.randint(0, num_motions, diversity_times))
    for first_idx, second_idx in zip(first_indices, second_indices):
        # TODO: 验证 torch.dist 是否等于 ms.ops.dist -> 解决
        diversity += ops.dist(activations[first_idx, :],
                              activations[second_idx, :])
    diversity /= diversity_times

    multimodality = 0
    label_quotas = np.repeat(multimodality_times, num_labels)
    while np.any(label_quotas > 0):
        # print(label_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not label_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        label_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += ops.dist(first_activation,
                                  second_activation)

    multimodality /= (multimodality_times * num_labels)

    return diversity.numpy().item(), multimodality.numpy().item()
