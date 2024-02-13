_base_ = 'default.py'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
		type='AAGCN',
        num_person=1,
        graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
