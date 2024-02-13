# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: graphconv.py.py
@Date: 2023/4/27 14:54
@Author: caijianfeng
"""
import math

import mindspore as ms

from mindspore import Parameter
from mindspore.nn import Cell


class GraphConvolution(Cell):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # TODO: 实现 mindspore 的 FloatTensor
        if bias:
            # self.bias = Parameter(torch.FloatTensor(out_features))
            pass
            # TODO: 实现 mindspore 的 FloatTensor
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = ms.ops.matmul(input, self.weight)
        # output = torch.spmm(adj, support)
        # spmm: 稀疏矩阵乘法
        # TODO: 验证
        output = ms.ops.csr_mul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
