# The based unit of graph convolutional networks.

import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor, ops
import numpy.random


class unit_gcn(nn.Cell):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: mindspore.Tensor,
                 adaptive: str = 'importance'):
        super().__init__()

        self.num_subsets = A.shape[0]
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * A.shape[0],
            1,
            has_bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.adaptive = adaptive
        self.A = Parameter(A)
        self.PA = Parameter(A)

    def construct(self, x):
        #  这里输入x是(N,C,T,V),经过conv(x)之后变为（N，C*kneral_size,T,V）
        n, c, t, v = x.shape

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({
                'offset': self.A + self.PA,
                'importance': self.A * self.PA
            })
        A = A_switch[self.adaptive]

        x = self.conv(x)

        x = x.view(n, self.num_subsets, -1, t, v)

        # x = torch. ('nkctv,kvw->nctw', (x, A)) 不支持，替换为普通算子
        n_dim, k_dim, c_dim, t_dim, v_dim = x.shape
        _, _, w_dim = A.shape
        reshape = ops.Reshape() # 变形算子
        mul = ops.Mul() # 乘法算子
        x = reshape(x, (n_dim, k_dim, c_dim, t_dim, v_dim, 1))
        A_t = reshape(A, (1, k_dim, 1, 1, v_dim, w_dim))
        x = mul(x, A_t)
        x = x.sum(axis=4)
        x = x.sum(axis=1)

        return self.act(self.bn(x))

class unit_tcn(nn.Cell):
    """The basic unit of temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the temporal convolution kernel.
            Defaults to 9.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        dilation (int): Spacing between temporal kernel elements.
            Defaults to 1.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        dropout (float): Dropout probability. Defaults to 0.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        norm: str = 'BN',
        dropout: float = 0.) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, pad, 0, 0),
            pad_mode='pad',
            stride=(stride, 1),
            dilation=(dilation, 1),
            has_bias=True)
        self.bn = nn.BatchNorm2d(out_channels) \
            if norm is not None else nn.Identity()

        self.drop = nn.Dropout(p=dropout)
        self.stride = stride

    def construct(self, x):
        """Defines the computation performed at every call."""
        return self.drop(self.bn(self.conv(x)))

if __name__=="__main__":
    A = numpy.random.random([3,17,17])
    A = Tensor(A, dtype=mindspore.float32)
    gcn = unit_gcn(3, 64, A)
    #  设 N=256*2, C=3, T=150, V=18
    shape = (256*2, 3, 150, 17)
    uniformreal = mindspore.ops.UniformReal(seed=2)
    x = uniformreal(shape)
    x = gcn(x)
    print(x.shape)


