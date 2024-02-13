# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: tensors.py
@Date: 2023/4/30 11:37
@Author: caijianfeng
"""
import mindspore as ms
import mindspore.ops as ops

def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = ops.arange(max_len)
    # ops.BroadcastTo = torch.Tensor.expand
    mask = ops.BroadcastTo((len(lengths), max_len.numpy().item()))(mask) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.shape[i] for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)  # (20, 60, 25, 6)
    # canvas = batch[0].new_zeros(size=size)
    canvas = ops.zeros(size, batch[0].dtype)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]  # (60, 25, 6)
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.shape[d])
        canvas[i] = sub_tensor.add(b)
    # print(canvas)
    return canvas


def collate(batch):
    # databatch = [b[0] for b in batch]
    # labelbatch = [b[1] for b in batch]
    # lenbatch = [len(b[0][0][0]) for b in batch]
    databatch = [b for b in batch[0]]
    labelbatch = batch[1]
    lenbatch = [len(b[0][0]) for b in databatch]

    databatchTensor = collate_tensors(databatch)
    # labelbatchTensor = ms.Tensor(labelbatch, dtype=ms.int64)
    labelbatchTensor = labelbatch
    # print(labelbatchTensor)
    lenbatchTensor = ms.Tensor(lenbatch, dtype=ms.int32)
    # print(lenbatchTensor)
    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    # print(maskbatchTensor)
    batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
    return batch
