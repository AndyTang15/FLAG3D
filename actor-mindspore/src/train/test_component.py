# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: test_component.py
@Date: 2023/5/1 11:03
@Author: caijianfeng
"""
from ..datasets.get_dataset import get_datasets
from src.parser.training import parser
from mindspore.dataset import GeneratorDataset
from src.utils.tensors import collate
import mindspore as ms
import mindspore.nn as nn
from ..models.get_model import get_model as get_gen_model

parameters = {'expname': 'exps',
              'folder': 'exps/humanact12',
              'batch_size': 1,
              'num_epochs': 5,
              'lr': 0.0001,
              'snapshot': 100,
              'dataset': 'humanact12',
              'num_frames': 60,
              'sampling': 'conseq',
              'sampling_step': 1,
              'pose_rep': 'rot6d',
              'max_len': -1, 'min_len': -1, 'num_seq_max': -1,
              'glob': True, 'glob_rot': [3.141592653589793, 0, 0],
              'translation': True, 'debug': False,
              'modelname': 'cvae_transformer_rc_rcxyz_kl',
              'latent_dim': 256,
              'lambda_kl': 1e-05, 'lambda_rc': 1.0, 'lambda_rcxyz': 1.0,
              'jointstype': 'vertices', 'vertstrans': False,
              'num_layers': 8, 'activation': 'gelu',
              'modeltype': 'cvae', 'archiname': 'transformer',
              'losses': ['rc', 'rcxyz', 'kl'],
              'lambdas': {'rc': 1.0, 'rcxyz': 1.0, 'kl': 1e-05},
              'num_classes': 12, 'nfeats': 6, 'njoints': 25}

ms.context.set_context(device_target='CPU')
datasets = get_datasets(parameters)
print('dataset construct success')
dataset = datasets['train']
train_iterator = GeneratorDataset(source=dataset, column_names=['data', 'label'])
train_iterator = train_iterator.shuffle(buffer_size=len(dataset))
train_iterator = train_iterator.batch(batch_size=parameters["batch_size"], num_parallel_workers=8)
# for data in train_iterator:
#     # print(type(data)) -> list
#     # print(data[0].shape, ';', data[1])
#     # print(data[0].dtype, ';', data[1].dtype)
#     data = collate(data)
#     data = {key: val for key, val in data.items()}
#     print(data['x'].shape, ';', data['y'].shape, ';', data['mask'].shape, ';', data['lengths'].shape)
#     break

model = get_gen_model(parameters)
print('model constrcut success')

# 在 batch 通过 model 后, batch 中用于计算 loss 的包括:
# batch["x"], batch["output"], batch["mask"] -> rc_loss
# batch["x_xyz"], batch["output_xyz"], batch["mask"] -> rcxyz_loss
# batch["mu"], batch["logvar"] -> kl_loss
mode = 'train'
for batch in train_iterator:
    batch = collate(batch)
    batch = {key: val for key, val in batch.items()}
    if mode == "train":
        optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=parameters["lr"])
        optimizer.requires_grad = True
    batch = model(batch)  # batch 更新
    print(batch["x"].shape,
          batch["output"].shape,
          batch["mask"].shape,
          batch["x_xyz"].shape,
          batch["output_xyz"].shape,
          batch["mu"].shape,
          batch["logvar"].shape)
    break
print('end')