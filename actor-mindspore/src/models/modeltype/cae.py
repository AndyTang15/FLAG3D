# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: cae.py
@Date: 2023/4/27 14:43
@Author: caijianfeng
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz


class CAE(nn.Cell):
    def __init__(self, encoder, decoder, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, device=None, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.outputxyz = outputxyz

        self.lambdas = lambdas

        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans

        self.losses = list(self.lambdas) + ["mixed"]

        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}

    def rot2xyz(self, x, mask, **kwargs):
        # for CVAE: x = batch["x"], mask = batch["mask"], kwargs = None
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, **kargs)  # shape = [batch size, ?, 3, 60]

    def construct(self, batch):
        if self.outputxyz:  # 进入
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])  # shape = [batch size, ?, 3, 60]
        elif self.pose_rep == "xyz":  # 不会进入
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))  # batch 多了 {"mu": mu, "logvar": logvar}
        # decode
        batch.update(self.decoder(batch))
        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def compute_loss(self, batch):
        mixed_loss = 0
        losses = {}
        # self.lambdas: {'rc': 1.0, 'rcxyz': 1.0, 'kl': 1e-05}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss * lam
            losses[ltype] = loss.numpy().item()
        losses["mixed"] = mixed_loss.numpy().item()
        return mixed_loss, losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, ms.Tensor):
            max_len = max_len.numpy().item()
        index = ops.arange(max_len).astype(ms.int32).expand(ms.Tensor((len(lengths), max_len), dtype=ms.int32))  # shape = (nspa * nats, max_len)
        # print(index.shape)
        mask = index < lengths.unsqueeze(1)
        return mask  # shape = (nspa * nats, max_len)

    def generate_one(self, cls, duration, fact=1, xyz=False):
        y = ms.Tensor([cls], ms.int32)[None]
        lengths = ms.Tensor([duration], ms.int32)
        mask = self.lengths_to_mask(lengths)
        z = ops.standard_normal(self.latent_dim)[None]

        batch = {"z": fact * z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if not xyz:
            return batch["output"][0]

        output_xyz = self.rot2xyz(batch["output"], batch["mask"])

        return output_xyz[0]

    def generate(self, classes, durations, nspa=1,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1):
        if nspa is None:
            nspa = 1
        nats = len(classes)

        y = classes.repeat(nspa)  # (view(nspa, nats)) -> shape = (nspa * nats, )

        if len(durations.shape) == 1:  # 进入
            lengths = durations.repeat(nspa)  # shape = (nspa * nats, )
        else:
            lengths = durations.reshape(y.shape)

        mask = self.lengths_to_mask(lengths)  # shape = (nspa * nats, max_len)

        if noise_same_action == "random":  # 进入
            if noise_diff_action == "random":  # 进入
                z = ops.standard_normal((nspa * nats, self.latent_dim))  # shape = (nspa * nats, 256)
            elif noise_diff_action == "same":
                z_same_action = ops.standard_normal((nspa, self.latent_dim))
                z = z_same_action.repeat_interleave(nats, axis=0)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
        elif noise_same_action == "interpolate":
            if noise_diff_action == "random":
                z_diff_action = ops.standard_normal((nats, self.latent_dim))
            elif noise_diff_action == "same":
                z_diff_action = ops.standard_normal((1, self.latent_dim))
                z_diff_action = ops.tile(z_diff_action, (nats, 1))
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            start = ms.Tensor(-1, ms.int32)
            stop = ms.Tensor(1, ms.int32)
            interpolation_factors = ops.linspace(start, stop, nspa)
            # z = torch.einsum("ij,k->kij", z_diff_action, interpolation_factors).view(nspa * nats, -1)
            # TODO: 实现 mindspore 的 einsum
        elif noise_same_action == "same":
            if noise_diff_action == "random":
                z_diff_action = ops.standard_normal(nats, self.latent_dim)
            elif noise_diff_action == "same":
                z_diff_action = ops.standard_normal(1, self.latent_dim).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            z = z_diff_action.repeat((nspa, 1))
        else:
            raise NotImplementedError("Noise same action must be random, same or interpolate.")
        # z.shape = (nspa * nats, 256); y.shape = (nspa * nats, );
        # mask.shape = (nspa * nats, max_len); lengths.shape = (nspa * nats, )
        batch = {"z": fact * z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)  # z.shape = (1, nspa * nats, 256); output.shape = (nspa * nats, 25, 6, max_len)
        if self.outputxyz:  # 进入
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])  # shape = [batch size, ?, 3, max_len]
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def return_latent(self, batch, seed=None):
        return self.encoder(batch)["z"]
