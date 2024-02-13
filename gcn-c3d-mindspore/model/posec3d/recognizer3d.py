

import mindspore
import mindspore.nn as nn

from model.posec3d.i3d_head import I3DHead
from model.posec3d.resnet3d_slowonly import ResNet3dSlowOnly


class Recognizer3d(nn.Cell):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 depth=50,
                 pretrained=None,
                 in_channels=17,
                 base_channels=32,
                 num_stages=3,
                 out_indices=(2,),
                 stage_blocks=(4, 6, 3),
                 conv1_stride_s=1,
                 pool1_stride_s=1,
                 inflate=(0, 1, 1),
                 spatial_strides=(2, 2, 2),
                 temporal_strides=(1, 1, 2),
                 dilations=(1, 1, 1),
                 num_classes=60,
                 in_channels_head=512,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01):
        super().__init__()

        self.backbone = ResNet3dSlowOnly(
        depth=depth,
        pretrained=pretrained,
        in_channels=in_channels,
        base_channels=base_channels,
        num_stages=num_stages,
        out_indices=out_indices,
        stage_blocks=stage_blocks,
        conv1_stride_s=conv1_stride_s,
        pool1_stride_s=pool1_stride_s,
        inflate=inflate,
        spatial_strides=spatial_strides,
        temporal_strides=temporal_strides,
        dilations=dilations
        )
        self.head = I3DHead(num_classes=num_classes,
                            in_channels_head=in_channels_head,
                            spatial_type=spatial_type,
                            dropout_ratio=dropout_ratio
                            )



    def construct(self, x):
        # B batch_size
        # N 视频个数
        # T = 100 一个视频的帧数paper规定是100帧，不足的重头循环，多的clip
        # V 17 根据不同的skeleton获得的节点数而定
        # M = 1 人数，paper中将人数限定在最大1个人

        # B N V T H W to B*M V T H W
        B, N, V, T, H, W = x.shape
        x = x.view(B*N, V, T, H, W)

        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = Recognizer3d(
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2,),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1),
        num_classes=60,
        in_channels_head=512,
        spatial_type='avg',
        dropout_ratio=0.5
    )
    # b, num_clip(double), num_keypoint(in_channel), frame, h, w
    shape = (4, 2, 17, 500, 56, 56)
    uniformreal = mindspore.ops.UniformReal(seed=2)
    x = uniformreal(shape)
    y = model(x)
    print(y.shape) # (B*M, 60)