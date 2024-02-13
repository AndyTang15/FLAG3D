import os
import sys
sys.path.append('./')
import mindspore
from mindspore import ops, nn
from logs.logger import get_logger
from evaluation.evaluation import top_k_accuracy
from model.posec3d.recognizer3d import Recognizer3d
from dataset import FLAG2DPoseC3DValDatasetGeneratorSingleGPU
import mindspore.dataset as ds

celoss = nn.CrossEntropyLoss()

def val(dataset_val, model, logger, batch_size):
    i = 0  # iteration num
    total_acc_top1 = 0.
    total_acc_top5 = 0.
    accuracy_num_top1 = 0.

    model.set_train(False)
    for data in dataset_val.create_dict_iterator():
        x = data["keypoint"]
        label = data["label"].to(mindspore.int32)
        y = model(x)
        loss = celoss(y, label)
        batch_acc = top_k_accuracy(y.numpy(), label.numpy(), (1,5))

        total_acc_top1 += batch_acc[0]
        total_acc_top5 += batch_acc[1]
        accuracy_num_top1 = accuracy_num_top1 + batch_acc[0] * batch_size # 4为batch_size

        logger.info("loss: " + str(loss) + " acc: " + str(batch_acc) + " label: " + str(label) + " pre_label: " + str(ops.argmax(y, dim=1)))

        i += 1


    accuracy_top1 = total_acc_top1 / i
    accuracy_top5 = total_acc_top5 / i
    logger.info("accuracy_num_top1: " + str(accuracy_num_top1))
    logger.info("accuracy_top1: " + str(accuracy_top1)  + " accuracy_top5: " + str(accuracy_top5))

mindspore.set_context(device_target="GPU")

batch_size = 4
step = 0
size =300

logs_path = "./exp"
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
logger = get_logger(logs_path + "/posec3d_flag2d_test_exp_["+str(step*size)+", "+str((step+1)*size)+").txt")

dataset_val_generator = FLAG2DPoseC3DValDatasetGeneratorSingleGPU("../data/FLAG/flag2d.pkl", step=step, size=size, clip_len = 500)
dataset_val = ds.GeneratorDataset(dataset_val_generator, ["keypoint", "label"],shuffle=True).batch(
        batch_size, True)
print("dataset success")

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
print("model success")
# for param in model.trainable_params():
#     print(param)
model.set_train(False)

param_dict = mindspore.load_checkpoint("./chpk_resume/posec3d_2d.ckpt")
# for i in param_dict:
#     print(param_dict[i])
mindspore.load_param_into_net(model, param_dict)
print("chpk success")

logger.info("posec3d_2d_val test for ["+str(step*size)+", "+str((step+1)*size)+")")
val(dataset_val, model, logger, batch_size)
print("val success")
# 备注：因单卡容量不足，需对数据集分批测试，需手动更改 step 参数，
# 对于FLAG2D， step = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
# 24 个 step 的 正确样本数 ：[260.0, 243.0, 192.0, 203.0, 256.0, 237.0, 220.0, 201.0, 244.0, 286.0, 262.0, 180.0, 220.0, 268.0, 239.0, 247.0, 252.0, 257.0, 232.0, 263.0, 246.0, 261.0, 243.0, 240.0]
# python ./test/PoseC3D_2d.py
