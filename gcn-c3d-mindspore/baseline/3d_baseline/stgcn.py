_base_ = 'default.py'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        num_person=1,
        graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
