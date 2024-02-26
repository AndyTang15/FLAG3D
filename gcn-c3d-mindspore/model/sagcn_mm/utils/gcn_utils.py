
import mindspore
import numpy as np
from mindspore import Parameter, Tensor, ParameterTuple, ops, nn

class unit_aagcn(nn.Cell):
    """The graph convolution unit of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_joints, num_joints)`.
        coff_embedding (int): The coefficient for downscaling the embedding
            dimension. Defaults to 4.
        adaptive (bool): Whether to use adaptive graph convolutional layer.
            Defaults to True.
        attention (bool): Whether to use the STC-attention module.
            Defaults to True.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1,
                     override=dict(type='Constant', name='bn', val=1e-6)),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out'),
                dict(type='ConvBranch', name='conv_d')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: mindspore.Tensor,
        coff_embedding: int = 4,
        adaptive: bool = True,
        attention: bool = True) -> None:

        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention

        num_joints = A.shape[-1]

        self.conv_d = nn.CellList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1, has_bias=True))

        if self.adaptive:
            self.A = Parameter(A)
            zeros = ops.Zeros()

            self.alpha = Parameter(zeros((1), mindspore.float32))
            self.conv_a = nn.CellList()
            self.conv_b = nn.CellList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1, has_bias=True))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1, has_bias=True))

        if self.attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4, pad_mode='pad', has_bias=True)
            # s attention
            ker_joint = num_joints if num_joints % 2 else num_joints - 1
            pad = (ker_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_joint, padding=pad, pad_mode='pad', has_bias=True)
            # channel attention
            rr = 2
            self.fc1c = nn.Dense(out_channels, out_channels // rr)
            self.fc2c = nn.Dense(out_channels // rr, out_channels)

        self.down = lambda x: x
        if in_channels != out_channels:
            self.down = nn.SequentialCell(
                nn.Conv2d(in_channels, out_channels, 1, has_bias=True),
                nn.BatchNorm2d(out_channels))

        self.matmul = ops.MatMul()
        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """Defines the computation performed at every call."""
        N, C, T, V = x.shape

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).view(
                    N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)

                bx, hx, wx = A1.shape
                by, hy, wy = A2.shape
                A1A2Mul = self.matmul(A1[0], A2[0])
                for j in range(1, bx):
                    A1A2Mul = ops.concat((A1A2Mul, self.matmul(A1[j], A2[j])), 0)
                A1A2Mul = A1A2Mul.view(bx, hx, wy)

                A1 = self.tan(A1A2Mul / A1.shape[-1])  # N V V

                ##################################################################
                A1 = self.A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)

                bx, hx, wx = A2.shape
                by, hy, wy = A1.shape
                # mindspore中矩阵乘法不支持三维，需自己实现，把第0维当作batch_size
                A2A1Mul = self.matmul(A2[0], A1[0])
                for k in range(1, bx):
                    A2A1Mul = ops.concat((A2A1Mul, self.matmul(A2[k], A1[k])))
                A2A1Mul = A2A1Mul.view(bx, hx, wy)

                z = self.conv_d[i](A2A1Mul.view(N, C, T, V))


                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](mindspore.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z


        y = self.relu(self.bn(y) + self.down(x))

        if self.attention:
            # spatial attention first
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
            y = y * se1.unsqueeze(-2) + y
            # then temporal attention
            se = y.mean(-1)  # N C T
            se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
            y = y * se1.unsqueeze(-1) + y
            # then spatial temporal attention ??
            se = y.mean(-1).mean(-1)  # N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))  # N C
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # A little bit weird
        return y


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
    A = np.random.random([3,17,17])
    A = Tensor(A, dtype=mindspore.float32)
    gcn = unit_aagcn(3, 64, A)
    # for para in gcn.parameters_dict():
    #     print(para)
    shape = (2, 3, 100, 17)
    uniformreal = mindspore.ops.UniformReal(seed=2)
    x = uniformreal(shape)
    y = gcn(x)
    print(y.shape)

