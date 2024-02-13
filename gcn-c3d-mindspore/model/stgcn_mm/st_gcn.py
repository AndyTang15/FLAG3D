import mindspore
import mindspore.nn as nn
import numpy

from mindspore import Parameter, Tensor, ParameterTuple, ops

from .utils.tgcn import unit_gcn
from .utils.tgcn import unit_tcn
from .utils.graph import Graph

class STGCN(nn.Cell):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_frames (int): Number of frames for the single video
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, graph_args,
                 num_stages: int = 10):
        super().__init__(auto_prefix=True)

        # load graph
        self.graph = Graph(**graph_args)
        A = Tensor(self.graph.A, dtype=mindspore.float32)
        # build networks
        spatial_kernel_size = A.shape[0]
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size) # (9,3)

        # 只对C归一化
        self.data_bnc = nn.BatchNorm1d(in_channels * A.shape[1])

        self.num_stages = num_stages
        self.gcn = nn.CellList([
            STGCNBlock(in_channels, 64, A, 1, residual=False),
            STGCNBlock(64, 64, A, 1),
            STGCNBlock(64, 64, A, 1),
            STGCNBlock(64, 64, A, 1),
            STGCNBlock(64, 128, A, 1),
            STGCNBlock(128, 128, A, 1),
            STGCNBlock(128, 128, A, 1),
            STGCNBlock(128, 256, A, 1),
            STGCNBlock(256, 256, A, 1),
            STGCNBlock(256, 256, A, 1),
        ])

    def construct(self, x):
        # B batch_size
        # N 视频个数
        # C = 3(X, Y, S) 代表一个点的信息(位置 + 预测的可能性)
        # T = 100 一个视频的帧数paper规定是100帧，不足的重头循环，多的clip
        # V 17 根据不同的skeleton获得的节点数而定
        # M = 1 人数，paper中将人数限定在最大1个人

        # B, N, M, T, V, C to N, C, T, V, M
        B, N, M, T, V, C = x.shape
        x = x.view(B * N, M, T, V, C)
        N = B * N
        x = x.permute(0, 1, 3, 4, 2)# N, M, T, V, C -> N, M, V, C, T

        # data normalization
        x = x.view(N * M, V * C, T) # (2*1, 3*17, 100)
        # 只对C归一化
        x = x.transpose((0, 2, 1)) # N * M, T, V * C
        x = self.data_bnc(x.view(N * M * T, V * C)).view(N * M, T, V * C)
        x = x.transpose((0, 2, 1)) # N * M, V * C, T

        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2)#.contiguous()
        x = x.view(N * M, C, T, V) # human1_video=[0:num_clip], human2_video=[num_clip:2*num_clip]

        # forwad
        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])

        return x #(bacth_size, num_class)

class STGCNBlock(nn.Cell):
    """The basic block of STGCN.

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


        self.gcn = unit_gcn(in_channels, out_channels, A)

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
        res = self.residual(x)
        x = self.tcn(self.gcn(x)) + res
        return self.relu(x)


if __name__=="__main__":
    # model测试
    model = STGCN(3, dict(layout='coco', mode='stgcn_spatial'))
    # for para in model.parameters_dict():
    #     print(para)
    shape = (1, 1, 1, 500, 17, 3) # dataloader直接读取的格式
    uniformreal = mindspore.ops.UniformReal(seed=2)
    x = uniformreal(shape)
    y = model(x)
    print(y.shape)
    # (2, 1, 1, 500, 17, 3) -> (2, 1, 256, 500, 17)
    # (2, 2, 1, 500, 17, 3) -> (4, 1, 256, 500, 17)

    # # stgcn测试
    # st_gcn = st_gcn(3, 64, (9, 1), 1)
    # #  整个网络的输入是一个(N = batch_size,C = 3,T = 300,V = 18,M = 2)的tensor。
    # #  设 N*M(2*2)/C(3)/T(150)/V(18)
    # shape = (4, 3, 150, 18)
    # uniformreal = mindspore.ops.UniformReal(seed=2)
    # x = uniformreal(shape)
    # A = numpy.random.rand(1, 18, 18)#Graph()
    # A = Parameter(Tensor(A, dtype=mindspore.float32), requires_grad=False)
    # x, A = st_gcn(x, A)
    # print(x.shape)