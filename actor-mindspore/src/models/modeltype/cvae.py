# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: cvae.py
@Date: 2023/4/27 14:52
@Author: caijianfeng
"""
import mindspore as ms
import mindspore.ops as ops
from .cae import CAE


class CVAE(CAE):
    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = ops.exp(logvar / 2)

        if seed is None:
            # eps = std.data.new(std.size()).normal_()
            randn = ops.UniformReal()
            eps = randn(std.shape)
        else:
            # generator = torch.Generator(device=self.device)
            # TODO: 实现 mindspore 的 Generator
            # generator.manual_seed(seed)
            ms.set_seed(seed)
            # torch 中的 std.data.new().normal_() 是生成一个正态分布矩阵(shape = std.shape)
            # eps = std.data.new(std.size()).normal_(generator=generator)
            randn = ops.UniformReal()
            eps = randn(std.shape)
        z = eps.mul(std).add(mu)
        return z

    def construct(self, batch):
        # batch['x'].shape = [batch size, 25, 6, 60]; batch['y'].shape = [batch size, ];
        # batch['mask'].shape = [batch size, 60]; batch['lengths'].shape = [batch size, ]

        if self.outputxyz:  # 进入
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])  # shape = [batch size, ?, 3, 60]
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))  # batch 多了 {"mu": mu, "logvar": logvar}
        batch["z"] = self.reparameterize(batch)

        # decode
        batch.update(self.decoder(batch))

        # if we want to output xyz
        if self.outputxyz:  # 进入
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])  # shape = [batch size, ?, 3, 60]
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def return_latent(self, batch, seed=None):
        distrib_param = self.encoder(batch)
        batch.update(distrib_param)
        return self.reparameterize(batch, seed=seed)
