import argparse
import datetime
import os
import sys

import mindspore
from mindspore import ops

from dataset import FLAG2DTrainDatasetGenerator, FLAG2DTestDatasetGenerator
from evaluation.evaluation import top_k_accuracy
from logs.logger import get_logger

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from model.sagcn_mm.sacgn_recognizer import AAGCN_RECOGNIZER


def main(args: argparse, logger):
    # dataloader
    print("loading train generator ...")
    dataset_train_generator = FLAG2DTrainDatasetGenerator(args.data_path, args.num_frames)
    print("loading test generator ...")
    dataset_test_generator = FLAG2DTestDatasetGenerator(args.data_path, args.num_frames, args.test_num_clip)
    dataset_train = ds.GeneratorDataset(dataset_train_generator, ["keypoint", "label"], shuffle=True).batch(
        args.batch_size, True)
    dataset_test = ds.GeneratorDataset(dataset_test_generator, ["keypoint", "label"], shuffle=True).batch(
        args.batch_size, True)

    # model
    model = AAGCN_RECOGNIZER(
        in_channels=args.in_channels,
        num_person = args.num_person,
        graph_args=args.graph_args,
        num_classes=args.num_class,
        in_channels_head=args.in_channels_head,
        kernel_size=args.kernel_size
    )


    # loss
    celoss = nn.CrossEntropyLoss()

    # scheduler
    iter_num = min(args.iter, len(dataset_train)//args.batch_size)
    lr_scheduler = nn.cosine_decay_lr(args.min_lr, args.max_lr, args.epochs * iter_num, iter_num, args.epochs)

    # optimizer
    optimizer = nn.SGD(params=model.trainable_params(),learning_rate=lr_scheduler,momentum=args.momentum,
                       weight_decay=args.weight_decay, nesterov=args.nesterov)

    if args.resume != None :
        param_dict = mindspore.load_checkpoint(args.resume)
        mindspore.load_param_into_net(model, param_dict)

    if args.mode == 'test':
        test_acc = test(dataset_test, model, celoss, logger, args)
        return


    if args.mode == 'train':
        best_acc = 0.
        for i in range(args.epochs):
            logger.info(f"Epoch {i}-------------------------------")
            train(dataset_train, model, celoss, optimizer, logger, args)
            val_acc = test(dataset_test, model, celoss, logger, args)

            if val_acc>best_acc:
                best_acc = val_acc
                mindspore.save_checkpoint(model, args.logs_path+"/"+"best.ckpt")
        return


def train(dataset_train, model, celoss, optimizer, logger, args):
    i=0 #iteration num
    total_loss = 0.
    model.set_train()


    def forward_fn(data, label):
        y = model(data)
        loss = celoss(y, label)
        return loss, y


    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    for data in dataset_train.create_dict_iterator():
        if(i>=args.iter):
            break
        x = data["keypoint"]
        label = data["label"].to(mindspore.int32)

        loss = train_step(x, label)
        logger.info("loss: " + str(loss))
        total_loss+=loss

        i += 1

    logger.info("train_total_avg_loss: " + str(total_loss/(args.batch_size*args.iter)))

def test(dataset_val, model, celoss, logger, args):
    i = 0  # iteration num
    total_loss = 0.
    total_acc = 0.

    model.set_train(False)
    for data in dataset_val.create_dict_iterator():
        x = data["keypoint"]
        label = data["label"].to(mindspore.int32)
        y = model(x)

        loss = celoss(y, label)

        batch_acc = top_k_accuracy(y.numpy(), label.numpy(), args.topk)

        logger.info("loss: " + str(loss) + " acc: " + str(batch_acc) + " label: " + str(label) + " pre_label: " + str(ops.argmax(y, dim=1)))

        total_acc += batch_acc[0]
        total_loss += loss

        i += 1

    accuracy = total_acc/ i
    logger.info("test_total_avg_loss: " + str(total_loss / (args.batch_size * i)) +  "accuracy_top1: " + str(accuracy))
    return accuracy


if __name__ == "__main__":
    logs_path = "./exp"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    logger = get_logger(logs_path + "/agcn_flag2d_train_exp.txt")

    parser = argparse.ArgumentParser(description='2sagcn for flag2d')
    # dataset parameter
    parser.add_argument('--data_path', default="../data/FLAG/flag2d.pkl", type=str,
                        help='where dataset locate')
    parser.add_argument('--logs_path', default=logs_path, type=str, help='where logs and ckpt locate')
    parser.add_argument('--resume', default=None, type=str, help='where trained model locate')
    # dataloader parameter
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--in_channels', default=3, type=int, help='Number of channels in the input data')
    parser.add_argument('--num_frames', default=500, type=int, help='Number of frames for the single video')
    parser.add_argument('--test_num_clip', default=1, type=int, help='Number of num_clip for the test dataset')
    parser.add_argument('--num_class', default=60, type=int, help='Number of classes for the classification task')
    parser.add_argument('--num_point', default=17, type=int, help='Number of keypoints for the classification task')
    parser.add_argument('--num_person', default=1, type=int, help='Number of person for the classification task')
    parser.add_argument('--graph_args', default=dict(layout='coco', mode='spatial'), type=dict,
                        help='The arguments for building the graph')
    parser.add_argument('--in_channels_head', default=256, type=int, help='in_channels_head')
    parser.add_argument('--kernel_size', default=(125, 17), type=dict,
                        help='kernel_size')

    # optimizer parameter
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov')
    # scheduler parameter
    parser.add_argument('--min_lr', default=0., type=float, help='min learning rate')
    parser.add_argument('--max_lr', default=0.8, type=float, help='max learning rate')
    # training parameter
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--iter', default=10000, type=int, help='iteration')
    parser.add_argument('--mode', default="train", type=str, help='train or test')
    # evaluate parameter
    parser.add_argument('--topk', default=(1,), type=dict, help='top k for evaluation')

    args = parser.parse_args()
    logger.info(args)
    mindspore.set_context(device_target="GPU")
    main(args, logger)

# python train_2sagcn_flag2d.py