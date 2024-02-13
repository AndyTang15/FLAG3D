# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: trainer.py
@Date: 2023/4/30 11:02
@Author: caijianfeng
"""
import mindspore as ms
import mindspore.ops as ops
from tqdm import tqdm
from src.utils.tensors import collate


def train_or_test(model, optimizer, iterator, mode="train"):
    if mode == "train":
        model.set_train()
        # grad_env = torch.enable_grad
    elif mode == "test":
        model.set_train(False)
        # grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    # loss of the epoch
    dict_loss = {loss: 0 for loss in model.losses}
    epoch_dict_loss = {loss: 0 for loss in model.losses}

    def forward_fn(data_x, data_y, data_mask, data_lengths):

        # batch = {
        #     "x": data_x,
        #     "output": data_output,
        #     "mask": data_mask,
        #     "x_xyz": data_x_xyz,
        #     "output_xyz": data_output_xyz,
        #     "mu": data_mu,
        #     "logvar": data_logvar
        # }
        batch = {
            "x": data_x,
            "y": data_y,
            "mask": data_mask,
            "lengths": data_lengths
        }
        batch = model(batch)
        mixed_loss, losses = model.compute_loss(batch)

        return mixed_loss, losses

    # has_aux=True 表示只有 forward_fn 的第一个输出贡献梯度
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # with grad_env():
    # iterator = tqdm(enumerate(iterator), desc="")
    for i, (inp, target) in tqdm(enumerate(iterator.create_tuple_iterator()), desc=f"Compute batch: "):
        # batch['x'].shape = [batch size, 25, 6, 60]; batch['y'].shape = [batch size, ];
        # batch['mask'].shape = [batch size, 60]; batch['lengths'].shape = [batch size, ]
        # inp, target = batch
        batch = (inp, target)
        # batch = [ms.Tensor.from_numpy(inp), ms.Tensor.from_numpy(target)]
        # Put everything in device
        batch = collate(batch)
        # batch: {'x': tensor(float),
        #         'y': tensor(int),
        #         'mask: tensor(bool),
        #         'lengths': tensor(int)}
        # batch = {key: val for key, val in batch.items()}

        # if mode == "train":
        #     # update the optimizer
        #     optimizer.requires_grad = True

        # forward pass
        # batch = model(batch)  # batch 更新
        # print(batch["x"].shape,
        #       batch["output"].shape,
        #       batch["mask"].shape,
        #       batch["x_xyz"].shape,
        #       batch["output_xyz"].shape,
        #       batch["mu"].shape,
        #       batch["logvar"].shape)
        # batch 中的参数:
        # batch["x"] = [batch size, 25, 6, 60]; batch["y"] = [batch size, ]
        # batch["output"] = [batch size, 25, 6, 60]
        # batch["mask"] = [batch size, 60]
        # batch["x_xyz"] = [batch size, 6890, 3, 60]
        # batch["output_xyz"] = [batch size, 6890, 3, 60]
        # batch["mu"] = [batch size, 256]
        # batch["logvar"] = [batch size, 256]
        # mixed_loss, losses = model.compute_loss(batch)
        # data_x, data_output, data_mask, data_x_xyz, data_output_xyz, data_mu, data_logvar = batch["x"], batch["output"], batch["mask"], batch["x_xyz"], batch["output_xyz"], batch["mu"], batch["logvar"]
        data_x, data_y, data_mask, data_lengths = batch["x"], batch["y"], batch["mask"], batch["lengths"]
        # print(data_x.dtype, '; ', data_x.shape, '\n', data_y.dtype, '; ', data_y.shape, '\n',
        #       data_mask.dtype, '; ', data_mask.shape, '\n', data_lengths.dtype, '; ', data_lengths.shape, '\n')

        # (mixed_loss, losses), grads = grad_fn(batch)
        (mixed_loss, losses), grads = grad_fn(data_x, data_y, data_mask, data_lengths)
        for key in dict_loss.keys():
            dict_loss[key] = losses[key]
            epoch_dict_loss[key] += losses[key]

        if mode == "train":
            # mixed_loss = ops.depend(mixed_loss, optimizer(grads))
            optimizer(grads)
            
            # iterator.desc = f"Computing batch: {i}; Loss: {mixed_loss}, train losses: {dict_loss}"

    return dict_loss, epoch_dict_loss


def train(model, optimizer, iterator):
    return train_or_test(model, optimizer, iterator, mode="train")


def test(model, optimizer, iterator):
    return train_or_test(model, optimizer, iterator, mode="test")
