import mindspore
from mindspore import Parameter, Tensor, ParameterTuple, ops, nn

from model.tsagcn_mm.utils.gcn_utils import unit_aagcn, unit_tcn
from .utils.graph import Graph


class AAGCNBlock(nn.Cell):
    """The basic block of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        residual (bool): Whether to use residual connection. Defaults to True.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: mindspore.Tensor,
                 stride: int = 1,
                 residual: bool = True) -> None:
        super().__init__()

        self.gcn = unit_aagcn(in_channels, out_channels, A)


        self.tcn = unit_tcn(
                out_channels, out_channels, 9, stride=stride)

        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """Defines the computation performed at every call."""
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))



class AAGCN(nn.Cell):
    """AAGCN backbone, the attention-enhanced version of 2s-AGCN.

    Skeleton-Based Action Recognition with Multi-Stream
    Adaptive Graph Convolutional Networks.
    More details can be found in the `paper
    <https://arxiv.org/abs/1912.06971>`__ .

    Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition.
    More details can be found in the `paper
    <https://arxiv.org/abs/1805.07694>`__ .

    Args:
        graph_cfg (dict): Config for building the graph.
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Number of base channels. Defaults to 64.
        data_bn_type (str): Type of the data bn layer. Defaults to ``'MVC'``.
        num_person (int): Maximum number of people. Only used when
            data_bn_type == 'MVC'. Defaults to 2.
        num_stages (int): Total number of stages. Defaults to 10.
        inflate_stages (list[int]): Stages to inflate the number of channels.
            Defaults to ``[5, 8]``.
        down_stages (list[int]): Stages to perform downsampling in
            the time dimension. Defaults to ``[5, 8]``.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 graph_args: dict,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 data_bn_type: str = 'MVC',
                 num_person: int = 1,
                 num_stages: int = 10,
                 inflate_stages: list = [5, 8],
                 down_stages: list = [5, 8]) -> None:
        super().__init__()

        self.graph = Graph(**graph_args)
        A = Tensor(self.graph.A, dtype=mindspore.float32)

        assert data_bn_type in ['MVC', 'VC', None]
        self.data_bn_type = data_bn_type
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_person = num_person
        self.num_stages = num_stages
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        if self.data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.shape[1])
        elif self.data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.shape[1])
        else:
            self.data_bn = nn.Identity()

        modules = []
        if self.in_channels != self.base_channels:
            modules = [
                AAGCNBlock(
                    in_channels,
                    base_channels,
                    A,
                    1,
                    residual=False)
            ]

        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(
                AAGCNBlock(
                    base_channels,
                    out_channels,
                    A,
                    stride=stride))
            base_channels = out_channels

        if self.in_channels == self.base_channels:
            self.num_stages -= 1

        self.gcn = nn.CellList(modules)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """Defines the computation performed at every call."""
        B, N, M, T, V, C = x.shape
        x = x.view(B * N, M, T, V, C)
        N = B * N
        x = x.permute(0, 1, 3, 4, 2)

        # batch norm
        x = x.view(N, M * V * C, T) # (2*1, 3*17, 100)
        # 只对C归一化
        x = x.transpose((0, 2, 1)) # N, T, M * V * C
        x = self.data_bn(x.view(N * T, M * V * C)).view(N , T , M * V * C)
        x = x.transpose((0, 2, 1))

        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4,
                                          2).view(N * M, C, T, V)
        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])

        return x

if __name__=="__main__":
    shape = (2, 1, 1, 100, 17, 3) # dataloader直接读取的格式
    uniformreal = mindspore.ops.UniformReal(seed=2)
    x = uniformreal(shape)
    model = AAGCN(
        graph_args=dict(layout='coco', mode='spatial'), num_person=1)
    y = model(x)
    print(y.shape)
    # (2, 1, 1, 100, 17, 3) -> (2, 1, 256, 25, 17)
