# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: accuracy.py
@Date: 2023/4/27 20:27
@Author: caijianfeng
"""
import mindspore as ms
import mindspore.ops as ops
import torch


def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        # TODO: mindspore 中如何显式不计算梯度 -> 解决
        for batch in motion_loader:
            output_xyz = torch.from_numpy(batch["output_xyz"].numpy())
            lengths = torch.from_numpy(batch["lengths"].numpy())
            batch_prob = classifier(output_xyz, lengths=lengths)
            batch_pred = batch_prob.max(dim=1).indices
            y = torch.from_numpy(batch["y"].numpy())
            for label, pred in zip(y, batch_pred):
                confusion[label][pred] += 1
    confusion = ms.Tensor.from_numpy(confusion.numpy())
    # print(confusion)
    accuracy = ms.Tensor.trace(confusion).float() / ops.reduce_sum(confusion).float()
    return accuracy.numpy().item(), confusion
