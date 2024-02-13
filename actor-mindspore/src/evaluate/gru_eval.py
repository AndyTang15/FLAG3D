# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: gru_eval.py
@Date: 2023/4/27 21:06
@Author: caijianfeng
"""
import mindspore as ms
from tqdm import tqdm

from src.utils.fixseed import fixseed

from src.evaluate.action2motion.evaluate import A2MEvaluation
# from src.evaluate.othermetrics.evaluation import OtherMetricsEvaluation

# from torch.utils.data import DataLoader
from mindspore.dataset import GeneratorDataset
from src.utils.tensors import collate

import os

from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model
from src.datasets.get_dataset import get_datasets


class NewDataloader:
    def __init__(self, mode, model, dataiterator, device=None):
        assert mode in ["gen", "rc", "gt"]
        self.batches = []
        # with torch.no_grad():
        model.set_train(False)
        for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
            databatch = collate(databatch)
            if mode == "gen":
                classes = databatch["y"]
                gendurations = databatch["lengths"]
                batch = model.generate(classes, gendurations)  # batch 中增加了 output 和 output_xyz
                # batch = {key: val.to(device) for key, val in batch.items()}
            elif mode == "gt":
                # batch = {key: val.to(device) for key, val in databatch.items()}
                batch = databatch
                batch["x_xyz"] = model.rot2xyz(batch["x"],  # shape = [batch size, ?, 3, 60]
                                               batch["mask"])
                batch["output"] = batch["x"]
                batch["output_xyz"] = batch["x_xyz"]
            elif mode == "rc":
                # databatch = {key: val.to(device) for key, val in databatch.items()}
                batch = model(databatch)
                batch["output_xyz"] = model.rot2xyz(batch["output"],  # shape = [batch size, ?, 3, 60]
                                                    batch["mask"])
                batch["x_xyz"] = model.rot2xyz(batch["x"],  # shape = [batch size, ?, 3, 60]
                                               batch["mask"])

            self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)


def evaluate(parameters, folder, checkpointname, epoch, niter):
    num_frames = 60

    # fix parameters for action2motion evaluation
    parameters["num_frames"] = num_frames
    if parameters["dataset"] == "ntu13":  # 不会进入
        parameters["jointstype"] = "a2m"
        parameters["vertstrans"] = False  # No "real" translation in this dataset
    elif parameters["dataset"] == "humanact12":  # 进入
        parameters["jointstype"] = "smpl"
        parameters["vertstrans"] = True
    else:
        raise NotImplementedError("Not in this file.")

    if "device" in parameters.keys():
        device = parameters["device"]
    else:
        device = None
    dataname = parameters["dataset"]

    # dummy => update parameters info
    get_datasets(parameters)  # get_datasets 是为了更新 parameters 参数, 增加 num_classes, nfeats, njoints
    model = get_gen_model(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = ms.load_checkpoint(checkpointpath)
    # model.load_state_dict(state_dict)
    param_not_load, _ = ms.load_param_into_net(model, state_dict)  # param_not_load 表示未能加载到model的参数(list)
    model.set_train(False)
    model.outputxyz = True

    a2mevaluation = A2MEvaluation(dataname)
    a2mmetrics = {}

    # evaluation = OtherMetricsEvaluation(device)
    # joints_metrics = {}, pose_metrics = {}

    datasetGT1 = get_datasets(parameters)["train"]
    datasetGT2 = get_datasets(parameters)["train"]

    allseeds = list(range(niter))  # 设置迭代次数(相当于 epoch), 每次会产生一个随机数进行 evaluate

    try:
        for index, seed in enumerate(allseeds):
            print(f"Evaluation number: {index+1}/{niter}")
            fixseed(seed)

            datasetGT1.reset_shuffle()  # reset_shuffle 表示没有 shuffle
            datasetGT1.shuffle()

            datasetGT2.reset_shuffle()
            datasetGT2.shuffle()
            # TODO
            # dataiterator = DataLoader(datasetGT1, batch_size=parameters["batch_size"],
            #                           shuffle=False, num_workers=8, collate_fn=collate)
            # TODO: 注意: 需要在后续读入数据时, 先将数据通过 collate -> 解决
            dataiterator = GeneratorDataset(source=datasetGT1, column_names=['data', 'label'])
            dataiterator = dataiterator.batch(batch_size=parameters["batch_size"], num_parallel_workers=8)
            # dataiterator2 = DataLoader(datasetGT2, batch_size=parameters["batch_size"],
            #                            shuffle=False, num_workers=8, collate_fn=collate)
            dataiterator2 = GeneratorDataset(source=datasetGT2, column_names=['data', 'label'])
            dataiterator2 = dataiterator2.batch(batch_size=parameters["batch_size"], num_parallel_workers=8)

            # reconstructedloader = NewDataloader("rc", model, dataiterator, device)
            motionloader = NewDataloader("gen", model, dataiterator, device)
            gt_motionloader = NewDataloader("gt", model, dataiterator, device)
            gt_motionloader2 = NewDataloader("gt", model, dataiterator2, device)

            # Action2motionEvaluation
            loaders = {"gen": motionloader,
                       # "recons": reconstructedloader,
                       "gt": gt_motionloader,
                       "gt2": gt_motionloader2}

            a2mmetrics[seed] = a2mevaluation.evaluate(model, loaders)

            # joints_metrics[seed] = evaluation.evaluate(model, num_classes,
            # loaders, xyz=True)
            # pose_metrics[seed] = evaluation.evaluate(model, num_classes,
            # loaders, xyz=False)

    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)

    metrics = {"feats": {key: [format_metrics(a2mmetrics[seed])[key] for seed in a2mmetrics.keys()] for key in a2mmetrics[allseeds[0]]}}
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}

    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)
