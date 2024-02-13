# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: losses.py
@Date: 2023/4/27 15:19
@Author: caijianfeng
"""
import mindspore as ms
# import torch.nn.functional as F
from .hessian_penalty import hessian_penalty
from .mmd import compute_mmd
import mindspore.ops as ops
import mindspore.nn as nn

def compute_rc_loss(model, batch):
    # batch['x'].shape = [batch size, 25, 6, 60];
    # batch["output"] = [batch size, 25, 6, 60];
    # batch["mask"] = [batch size, 60]
    x = batch["x"]
    output = batch["output"]
    mask = batch["mask"]

    gtmasked = x.permute(0, 3, 1, 2)[mask]  # [batch size, 60, 25, 6]
    # x = x.permute(0, 3, 1, 2).numpy()
    # mask = mask.numpy()
    # gtmasked = ms.Tensor.from_numpy(x[mask]).astype(ms.float32)

    outmasked = output.permute(0, 3, 1, 2)[mask]  # [batch size, 60, 25, 6]
    # output = output.permute(0, 3, 1, 2).numpy()
    # outmasked = ms.Tensor.from_numpy(output[mask]).astype(ms.float32)

    loss = nn.MSELoss(reduction='mean')(gtmasked, outmasked)
    return loss


def compute_rcxyz_loss(model, batch):
    # batch["x_xyz"] = [batch size, 6890, 3, 60];
    # batch["output_xyz"] = [batch size, 6890, 3, 60]
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    mask = batch["mask"]

    gtmasked = x.permute(0, 3, 1, 2)[mask]  # [batch size, 60, 6890, 3]
    # x = x.permute(0, 3, 1, 2).numpy()
    # mask = mask.numpy()
    # gtmasked = ms.Tensor.from_numpy(x[mask]).astype(ms.float32)

    outmasked = output.permute(0, 3, 1, 2)[mask]  # [batch size, 60, 6890, 3]
    # output = output.permute(0, 3, 1, 2).numpy()
    # outmasked = ms.Tensor.from_numpy(output[mask]).astype(ms.float32)

    # loss = F.mse_loss(gtmasked, outmasked, reduction='mean')
    loss = nn.MSELoss(reduction='mean')(gtmasked, outmasked)
    return loss


def compute_vel_loss(model, batch):
    x = batch["x"]
    output = batch["output"]
    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    mask = batch["mask"][..., 1:]

    gtvelmasked = gtvel.permute(0, 3, 1, 2)[mask]
    # gtvel = gtvel.permute(0, 3, 1, 2).numpy()
    # mask = mask.numpy()
    # gtvelmasked = ms.Tensor.from_numpy(gtvel[mask]).astype(ms.float32)

    outvelmasked = outputvel.permute(0, 3, 1, 2)[mask]
    # outputvel = outputvel.permute(0, 3, 1, 2).numpy()
    # outvelmasked = ms.Tensor.from_numpy(outputvel[mask]).astype(ms.float32)

    loss = nn.MSELoss(reduction='mean')(gtvelmasked, outvelmasked)
    return loss


def compute_velxyz_loss(model, batch):
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    mask = batch["mask"][..., 1:]

    gtvelmasked = gtvel.permute(0, 3, 1, 2)[mask]
    # gtvel = gtvel.permute(0, 3, 1, 2).numpy()
    # mask = mask.numpy()
    # gtvelmasked = ms.Tensor.from_numpy(gtvel[mask]).astype(ms.float32)

    outvelmasked = outputvel.permute(0, 3, 1, 2)[mask]
    # outputvel = outputvel.permute(0, 3, 1, 2).numpy()
    # outvelmasked = ms.Tensor.from_numpy(outputvel[mask]).astype(ms.float32)

    loss = nn.MSELoss(reduction='mean')(gtvelmasked, outvelmasked)
    return loss


def compute_hp_loss(model, batch):
    # TODO: 实现 torch.random.seed 的对应的 mindspore -> 解决
    loss = hessian_penalty(model.return_latent, batch, seed=ms.get_seed())
    return loss


def compute_kl_loss(model, batch):
    mu, logvar = batch["mu"], batch["logvar"]
    # loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = -0.5 * ops.reduce_sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss


def compute_mmd_loss(model, batch):
    z = batch["z"]
    true_samples = ops.standard_normal(z.shape)
    loss = compute_mmd(true_samples, z)
    return loss


_matching_ = {"rc": compute_rc_loss, "kl": compute_kl_loss, "hp": compute_hp_loss,
              "mmd": compute_mmd_loss, "rcxyz": compute_rcxyz_loss,
              "vel": compute_vel_loss, "velxyz": compute_velxyz_loss}


def get_loss_function(ltype):
    return _matching_[ltype]


def get_loss_names():
    return list(_matching_.keys())
