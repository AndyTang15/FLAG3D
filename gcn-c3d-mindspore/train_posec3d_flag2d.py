import argparse
import datetime
import os
import sys

import mindspore
from mindspore import ops

from dataset import FLAG2DPoseC3DTrainDatasetGenerator, FLAG2DPoseC3DValDatasetGenerator, FLAG2DPoseC3DTestDatasetGenerator
from evaluation.evaluation import top_k_accuracy
from logs.logger import get_logger
from model.posec3d.recognizer3d import Recognizer3d

import mindspore.dataset as ds
import mindspore.nn as nn

def main(args: argparse, logger):
    # dataloader
    print("loading train generator ...")
    dataset_train_generator = FLAG2DPoseC3DTrainDatasetGenerator(args.data_path, args.num_frames)
    print("loading val generator ...")
    dataset_val_generator = FLAG2DPoseC3DValDatasetGenerator(args.data_path, args.num_frames)
    print("loading test generator ...")
    dataset_test_generator = FLAG2DPoseC3DTestDatasetGenerator(args.data_path, args.num_frames, args.test_num_clip)
    dataset_train = ds.GeneratorDataset(dataset_train_generator, ["keypoint", "label"], shuffle=True).batch(
        args.batch_size, True)
    dataset_val = ds.GeneratorDataset(dataset_val_generator, ["keypoint", "label"], shuffle=True).batch(
        args.batch_size, True)
    dataset_test = ds.GeneratorDataset(dataset_test_generator, ["keypoint", "label"], shuffle=True).batch(
        args.batch_size, True)

    # model
    model = Recognizer3d(
        depth=args.depth,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        out_indices=args.out_indices,
        stage_blocks=args.stage_blocks,
        conv1_stride_s=args.conv1_stride_s,
        pool1_stride_s=args.pool1_stride_s,
        inflate=args.inflate,
        spatial_strides=args.spatial_strides,
        temporal_strides=args.temporal_strides,
        dilations=args.dilations,
        num_classes=args.num_class,
        in_channels_head=args.in_channels_head,
        spatial_type=args.spatial_type,
        dropout_ratio=args.dropout_ratio)

    # loss
    celoss = nn.CrossEntropyLoss()

    # scheduler
    lr_scheduler = nn.cosine_decay_lr(args.min_lr, args.max_lr, args.epochs * args.iter, args.iter, args.epochs)

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
            val_acc = val(dataset_val, model, celoss, logger, args)

            if val_acc>best_acc:
                best_acc = val_acc
                mindspore.save_checkpoint(model, args.logs_path+"/"+"best.ckpt")

        test(dataset_test, model, celoss, logger, args)

        return


def train(dataset_train, model, celoss, optimizer, logger, args):
    i=0 #iteration num
    total_loss = 0.
    model.set_train()

    # 定义前传函数
    def forward_fn(data, label):
        y = model(x)
        loss = celoss(y, label)
        return loss, y

    # 生成求导函数，用于计算给定函数的正向计算结果和梯度。
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

def val(dataset_val, model, celoss, logger, args):
    i = 0  # iteration num
    total_loss = 0.
    total_acc_top1 = 0.
    total_acc_top5 = 0.

    model.set_train(False)
    for data in dataset_val.create_dict_iterator():
        x = data["keypoint"]
        label = data["label"].to(mindspore.int32)
        y = model(x)
        # 求loss
        loss = celoss(y, label)
        # 求每个batch准确率
        batch_acc = top_k_accuracy(y.numpy(), label.numpy(), args.topk)

        logger.info("loss: " + str(loss) + " acc: " + str(batch_acc) + " label: " + str(label) + " pre_label: " + str(ops.argmax(y, dim=1)))

        total_acc_top1 += batch_acc[0]
        total_acc_top5 += batch_acc[1]
        total_loss += loss

        i += 1


    accuracy_top1 = total_acc_top1 / i
    accuracy_top5 = total_acc_top5 / i
    logger.info("val_total_avg_loss: " + str(total_loss / (args.batch_size * i)) +  "accuracy_top1: " + str(accuracy_top1) + "accuracy_top5: " + str(accuracy_top5))
    return accuracy_top1, accuracy_top5

def test(dataset_test, model, celoss, logger, args):
    i = 0  # iteration num
    total_loss = 0.
    total_acc_top1 = 0.
    total_acc_top5 = 0.

    model.set_train(False)
    for data in dataset_test.create_dict_iterator():
        x = data["keypoint"]
        label = data["label"].to(mindspore.int32)
        y = model(x)
        # 对10个clip求平均再softmax
        y = y.view(args.batch_size, args.test_num_clip, -1)
        y = ops.mean(y, 1, keep_dims=False)

        # 求loss
        loss = celoss(y, label)

        # 求每个batch准确率
        batch_acc = top_k_accuracy(y.numpy(), label.numpy(), args.topk)

        logger.info("loss: " + str(loss) + " acc: " + str(batch_acc) + " label: " + str(label) + " pre_label: " + str(
            ops.argmax(y, dim=1)))

        total_acc_top1 += batch_acc[0]
        total_acc_top5 += batch_acc[1]
        total_loss += loss

        i += 1

    accuracy_top1 = total_acc_top1 / i
    accuracy_top5 = total_acc_top5 / i
    print("val_total_avg_loss: ", total_loss / (args.batch_size * i),  "accuracy_top1: ", accuracy_top1 ,"accuracy_top5: ", accuracy_top5)
    return accuracy_top1, accuracy_top5

if __name__ == "__main__":
    logs_path = "./exp"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    logger = get_logger(logs_path + "/posec3d_flag2d_train_exp.txt")

    parser = argparse.ArgumentParser(description='posec3d for flag2d')
    # dataset parameter
    parser.add_argument('--data_path', default="../data/FLAG/flag2d.pkl", type=str,
                        help='where dataset locate')
    parser.add_argument('--logs_path', default=logs_path, type=str, help='where logs and ckpt locate')
    parser.add_argument('--resume', default='./chpk_resume/posec3d_2d.ckpt', type=str, help='where trained model locate')
    # dataloader parameter
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_frames', default=500, type=int, help='Number of frames for the single video')
    parser.add_argument('--test_num_clip', default=10, type=int, help='Number of num_clip for the test dataset')
    parser.add_argument('--num_class', default=60, type=int, help='Number of classes for the classification task')
    parser.add_argument('--num_point', default=17, type=int, help='Number of keypoints for the classification task')
    # model parameter
    parser.add_argument('--depth', default=50, type=int, help='resnet depth')
    parser.add_argument('--in_channels', default=17, type=int, help='Number of channels in the input data')
    parser.add_argument('--pretrained', default=None, help='is model pretrain?')
    parser.add_argument('--base_channels', default=32, type=int, help='model base_channel')
    parser.add_argument('--num_stages', default=3, type=int, help='resnet stage')
    parser.add_argument('--out_indices', default=(2,), type=dict, help='resnet out stage index')
    parser.add_argument('--stage_blocks', default=(4, 6, 3), type=dict, help='resnet stage blocks num')
    parser.add_argument('--conv1_stride_s', default=1, type=int, help='')
    parser.add_argument('--pool1_stride_s', default=1, type=int, help='')
    parser.add_argument('--inflate', default=(0, 1, 1), type=dict, help='')
    parser.add_argument('--spatial_strides', default=(2, 2, 2), type=dict, help='')
    parser.add_argument('--temporal_strides', default=(1, 1, 2), type=dict, help='')
    parser.add_argument('--dilations', default=(1, 1, 1), type=dict, help='')
    parser.add_argument('--in_channels_head', default=512, type=int, help='Number of channels in the input data for i3dhead')
    parser.add_argument('--spatial_type', default='avg', type=str, help='spatial_type for head')
    parser.add_argument('--dropout_ratio', default=0.5, type=int, help='dropout_ratio for head')


    # optimizer parameter
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=0.0003, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov')
    # scheduler parameter
    parser.add_argument('--min_lr', default=0., type=float, help='min learning rate')
    parser.add_argument('--max_lr', default=0.8, type=float, help='max learning rate')
    # training parameter
    parser.add_argument('--epochs', default=30, type=int, help='epochs')
    parser.add_argument('--iter', default=100, type=int, help='iteration')
    parser.add_argument('--mode', default="train", type=str, help='train or test')
    # evaluate parameter
    parser.add_argument('--topk', default=(1,5), type=dict, help='top k for evaluation')

    args = parser.parse_args()
    logger.info(args)
    mindspore.set_context(device_target="GPU")
    main(args, logger)

# python train_posec3d_flag2d.py