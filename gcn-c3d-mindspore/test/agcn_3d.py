import os
import sys
sys.path.append('./')
import mindspore
from mindspore import ops, nn
from logs.logger import get_logger
from evaluation.evaluation import top_k_accuracy
from dataset import FLAG3DValDatasetGenerator
import mindspore.dataset as ds

from model.sagcn_mm.sacgn_recognizer import AAGCN_RECOGNIZER

celoss = nn.CrossEntropyLoss()

def val(dataset_val, model, logger):
    i = 0  # iteration num
    total_acc = 0.

    #  MindSpore 只有在调用grad才会根据正向图结构来构建反向图，
    # 正向执行时不会记录任何信息，所以 MindSpore 并不需要该接口，
    # 也可以理解为 MindSpore 的正向计算均在torch.no_grad 情况下进行的。

    model.set_train(False)
    for data in dataset_val.create_dict_iterator():
        x = data["keypoint"]
        label = data["label"].to(mindspore.int32)
        y = model(x)
        loss = celoss(y, label)
        # 求loss
        # 求每个batch准确率
        batch_acc = top_k_accuracy(y.numpy(), label.numpy(), (1,))

        total_acc += batch_acc[0]

        logger.info("loss: " + str(loss) + " acc: " + str(batch_acc) + " label: " + str(label) + " pre_label: " + str(ops.argmax(y, dim=1)))

        i += 1

    accuracy = total_acc/ i
    logger.info("accuracy_top1: " + str(accuracy))
    return accuracy

mindspore.set_context(device_target="GPU")

logs_path = "./exp"
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
logger = get_logger(logs_path + "/agcn_flag3d_test_exp.txt")

dataset_val_generator = FLAG3DValDatasetGenerator("../data/FLAG/flag3d.pkl", 500)
dataset_val = ds.GeneratorDataset(dataset_val_generator, ["keypoint", "label"], shuffle=True).batch(
        4, True)
print("dataset success")

model = AAGCN_RECOGNIZER(
        in_channels=3,
        num_person = 1,
        graph_args=dict(layout='nturgb+d', mode='stgcn_spatial'),
        num_classes=60,
        in_channels_head=256,
        kernel_size=(125, 25)
    )
print("model success")
# for param in model.trainable_params():
#     print(param)
# for para in model.parameters_dict():
#     print(para)
model.set_train(False)

param_dict = mindspore.load_checkpoint("./chpk_resume/agcn_3d.ckpt")
mindspore.load_param_into_net(model, param_dict)
print("chpk success")

val(dataset_val, model, logger)
print("val success")

# python ./test/agcn_3d.py
