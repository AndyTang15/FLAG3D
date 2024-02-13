# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: transformer.py
@Date: 2023/4/27 14:32
@Author: caijianfeng
"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


# import torch.nn.functional as F


class PositionalEncoding(nn.Cell):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # d_model = 256
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = ops.zeros((max_len, d_model), ms.float32)  # shape = (5000, 256)
        position = ops.arange(0, max_len, dtype=ms.float32).unsqueeze(1)  # (5000, 1)
        div_term = ops.exp(ops.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # (128, )
        pe[:, 0::2] = ops.sin(position * div_term)  # (5000, 128) -> 切片: 从 0 开始, 间隔为 2
        pe[:, 1::2] = ops.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 0, 2)  # shape = (5000, 1, 256)

        self._set_attr_for_tensor('pe', pe)

    def construct(self, x):
        # not used in the final model
        # x.shape = [62, batch size, 256]; self.pe[:x.shape[0], :].shape = [62, 1, 256]
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)  # [62, batch size, 256]


# only for ablation / not used in the final model
class TimeEncoding(nn.Cell):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, mask, lengths):
        time = mask * 1 / (lengths[..., None] - 1)
        time = time[:, None] * ops.arange(time.shape[1])[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Cell):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot, batch_size,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()
        self.batch_size = batch_size
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        if self.ablation == "average_encoder":
            self.mu_layer = nn.Dense(self.latent_dim, self.latent_dim)
            self.sigma_layer = nn.Dense(self.latent_dim, self.latent_dim)
        else:
            randn = ops.StandardNormal()
            self.muQuery = ms.Parameter(randn((self.num_classes, self.latent_dim)), requires_grad=True)  # [12, 256]
            self.sigmaQuery = ms.Parameter(randn((self.num_classes, self.latent_dim)), requires_grad=True)  # [12, 256]

        self.skelEmbedding = nn.Dense(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # TODO: 设置 seq_length + batch_size
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(encoder_layer=seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

    def construct(self, batch):
        # batch['x'].shape = [batch size, 25, 6, 60];
        # batch['y'].shape = [batch size, ];
        # batch['mask'].shape = [batch size, 60].
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape  # batch size, 25, 6, 60
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)  # [60, batch size, 25*6]

        # embedding of the skeleton
        x = self.skelEmbedding(x)  # [60, batch size, 256]

        # only for ablation / not used in the final model
        if self.ablation == "average_encoder":  # 不会进入
            # add positional encoding
            x = self.sequence_pos_encoder(x)

            # transformer layers
            # TODO: 确认 torch.nn.TransformerEncoder 对应 mindspore 的 TransformerEncoder 的每个参数
            final = self.seqTransEncoder(src=x, src_key_padding_mask=~mask)
            # get the average of the output
            z = final.mean(axis=0)

            # extract mu and logvar
            mu = self.mu_layer(z)
            logvar = self.sigma_layer(z)
        else:
            # adding the mu and sigma queries
            xseq = ops.concat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)  # shape = [62, batch size, 256]

            # add positional encoding
            xseq = self.sequence_pos_encoder(xseq)  # [62, batch size, 256]

            # create a bigger mask, to allow attend to mu and sigma
            muandsigmaMask = ops.ones((bs, 2), ms.bool_)  # [batch size, 2]
            maskseq = ops.concat((muandsigmaMask, mask), axis=1)  # [batch size, 62]
            # TODO: 确认 torch.nn.TransformerDecoder 对应 mindspore 的 TransformerDecoder 的每个参数
            final = self.seqTransEncoder(src=xseq, src_key_padding_mask=~maskseq)  # [62, 20, 256]
            # 取前两个作为 预测得到的 均值与方差
            mu = final[0]  # shape = [batch size, 256]
            logvar = final[1]  # shape = [batch size, 256]

        return {"mu": mu, "logvar": logvar}


class Decoder_TRANSFORMER(nn.Cell):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation

        self.input_feats = self.njoints * self.nfeats  # 25 * 6

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            self.ztimelinear = nn.Dense(self.latent_dim + self.num_classes, self.latent_dim)
        else:
            randn = ops.StandardNormal()
            self.actionBiases = ms.Parameter(randn((self.num_classes, self.latent_dim)))  # (12, 256)

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(decoder_layer=seqTransDecoderLayer,
                                                     num_layers=self.num_layers)

        self.finallayer = nn.Dense(self.latent_dim, self.input_feats)

    def construct(self, batch):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]
        latent_dim = z.shape[1]  # 256
        bs, nframes = mask.shape  # batch size, 60
        njoints, nfeats = self.njoints, self.nfeats  # 25, 6

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":  # 不会进入
            yoh = ops.one_hot(y, self.num_classes, on_value=ms.Tensor(1, dtype=ms.float32),
                              off_value=ms.Tensor(0, dtype=ms.float32))
            z = ops.concat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]  # sequence of size 1
        else:  # 进入
            # only for ablation / not used in the final model
            if self.ablation == "concat_bias":  # 不会进入
                # sequence of size 2
                # z = torch.stack((z, self.actionBiases[y]), axis=0)
                z = ops.stack((z, self.actionBiases[y]), axis=0)
            else:  # 进入
                # shift the latent noise vector to be the action noise
                z = z + self.actionBiases[y]  # shape = [bs, 256]
                z = z[None]  # sequence of size 1 -> shape = [1, bs, 256]

        timequeries = ops.zeros((nframes, bs, latent_dim), ms.float32)  # shape = (60, batch size, 256)

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":  # 不会进入
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:  # 进入
            timequeries = self.sequence_pos_encoder(timequeries)  # (nframes=60, bs, latent_dim=256)
        # TODO: 确认 torch.nn.TransformerDecoder 对应 mindspore 的 TransformerDecoder 的每个参数
        output = self.seqTransDecoder(tgt=timequeries, memory=z,  # output.shape = (60, batch size, 256)
                                      tgt_key_padding_mask=~mask)
        # self.finallayer(output).shape = (60, bs, 25 * 6) -> reshape: (60, bs, 25, 6)
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)

        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)  # (bs, 25, 6, 60)

        batch["output"] = output
        return batch
