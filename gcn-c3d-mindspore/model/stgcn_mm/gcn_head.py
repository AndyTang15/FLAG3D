import mindspore
from mindspore import nn

class GCNHead(nn.Cell):
    """The classification head for GCN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss')``.
        dropout (float): Probability of dropout layer. Defaults to 0.
        init_cfg (dict or list[dict]): Config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels_head: int,
                 kernel_size: tuple,
                 dropout: float = 0.):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels_head
        self.dropout_ratio = dropout
        if self.dropout_ratio != 1.:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        # self.pool = nn.AdaptiveAvgPool2d(1) # for gpu
        self.pool = nn.AvgPool2d(pad_mode="VALID", kernel_size=kernel_size) # for cpu
        self.fc = nn.Dense(self.in_channels, self.num_classes)

    def construct(self, x):
        """Forward features from the upstream network.

        Args:
            x (torch.Tensor): Features from the upstream network.

        Returns:
            torch.Tensor: Classification scores with shape (B, num_classes).
        """
        N, M, C, T, V = x.shape
        x = x.view(N * M, C, T, V)
        x = self.pool(x)
        x = x.view(N, M, C)
        x = x.mean(axis=1)
        assert x.shape[1] == self.in_channels

        if self.dropout is not None:
            x = self.dropout(x)

        cls_scores = self.fc(x)
        return cls_scores

if __name__=="__main__":
    shape = (1, 1, 256, 500, 17)
    uniformreal = mindspore.ops.UniformReal(seed=2)
    x = uniformreal(shape)
    kernel_size = x.shape[3:]
    head = GCNHead(60, 256, kernel_size)
    y = head(x)
    print(y.shape)