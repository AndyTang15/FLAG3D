# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: accuracy.py
@Date: 2023/4/27 20:55
@Author: caijianfeng
"""
import mindspore as ms
import mindspore.ops as ops

def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = ops.zeros(num_labels, num_labels, dtype=ms.int64)
    # with torch.no_grad():
    # TODO: 验证 model.eval() 是否可以替代 torch.no_grad()
    model.eval()
    for batch in motion_loader:
        batch_prob = classifier(batch)["yhat"]
        batch_pred = batch_prob.max(dim=1).indices
        for label, pred in zip(batch["y"], batch_pred):
            confusion[label][pred] += 1

    accuracy = ms.Tensor.trace(confusion)/ops.reduce_sum(confusion)
    return accuracy.item(), confusion
