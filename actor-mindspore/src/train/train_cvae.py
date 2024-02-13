# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: train_cvae.py
@Date: 2023/4/29 13:55
@Author: caijianfeng
"""
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
# from torch.utils.tensorboard import SummaryWriter

# from torch.utils.data import DataLoader
from src.train.trainer import train
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data


def do_epochs(model, datasets, parameters, optimizer):
    dataset = datasets["train"]
    # dataset 的每个数据: (x: tensor, label: int)
    # print(dataset[0][0].shape)
    # dataset 的数据: x: tensor -> shape=[25, 6, 60]
    train_iterator = GeneratorDataset(source=dataset, column_names=['data', 'label'])
    train_iterator = train_iterator.shuffle(buffer_size=len(train_iterator))
    train_iterator = train_iterator.batch(batch_size=parameters["batch_size"], num_parallel_workers=8)
    # train_iterator = DataLoader(dataset, batch_size=parameters["batch_size"],
    #                             shuffle=True, num_workers=8, collate_fn=collate)
    print(f'dataset construct success, the total data batches are {len(train_iterator)}')
    logpath = os.path.join(parameters["folder"], "training.log")
    with open(logpath, "w") as logfile:
        check_point = 1000
        for epoch in range(check_point+1, parameters["num_epochs"] + 1):
            dict_loss, epoch_dict_loss = train(model, optimizer, train_iterator)

            # for key in dict_loss.keys():
            #     dict_loss[key] /= len(train_iterator)
            #     writer.add_scalar(f"Loss/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train final batch losses: {dict_loss}, train_epoch_loss: {epoch_dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)

            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"] + check_point):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.ckpt'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                ms.save_checkpoint(model, checkpoint_path)

            # writer.flush()


if __name__ == '__main__':
    # parse options
    parameters = parser()

    # logging tensorboard
    # writer = SummaryWriter(log_dir=parameters["folder"])
    # ms.context.set_context(device_target='CPU')
    ms.context.set_context(device_target='GPU')
    model, datasets = get_model_and_data(parameters)
    check_point_file = "./exps/humanact12/checkpoint_1000.ckpt"
    check_point_param = ms.load_checkpoint(check_point_file)
    ms.load_param_into_net(model, check_point_param)

    # optimizer
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=parameters["lr"])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.trainable_params()) / 1000000.0))
    print("Training model (check point is 1000)..")
    do_epochs(model, datasets, parameters, optimizer)

    # writer.close()
