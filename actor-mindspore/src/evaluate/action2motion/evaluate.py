# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: evaluate.py
@Date: 2023/4/27 20:33
@Author: caijianfeng
"""
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np
from .models import load_classifier, load_classifier_for_fid
from .accuracy import calculate_accuracy
from .fid import calculate_fid
from .diversity import calculate_diversity_multimodality


class A2MEvaluation:
    def __init__(self, dataname, device=None):
        dataset_opt = {"ntu13": {"joints_num": 18,
                                 "input_size_raw": 54,
                                 "num_classes": 13},
                       'humanact12': {"input_size_raw": 72,
                                      "joints_num": 24,
                                      "num_classes": 12}}

        if dataname not in dataset_opt.keys():
            assert NotImplementedError(f"{dataname} is not supported.")

        self.dataname = dataname
        self.input_size_raw = dataset_opt[dataname]["input_size_raw"]
        self.num_classes = dataset_opt[dataname]["num_classes"]
        if not device:
            device = torch.device('cpu')
        self.device = device

        self.gru_classifier_for_fid = load_classifier_for_fid(dataname, self.input_size_raw,
                                                              self.num_classes, self.device).eval()
        self.gru_classifier = load_classifier(dataname, self.input_size_raw,
                                              self.num_classes, self.device).eval()

    def compute_features(self, model, motionloader):
        # calculate_activations_labels function from action2motion
        activations = []
        labels = []
        with torch.no_grad():  # TODO
            for idx, batch in enumerate(motionloader):
                output_xyz = torch.from_numpy(batch["output_xyz"].numpy())
                lengths = torch.from_numpy(batch["lengths"].numpy())
                activations.append(ms.Tensor.from_numpy(self.gru_classifier_for_fid(output_xyz, lengths=lengths).numpy()))
                labels.append(batch["y"])
            activations = ops.concat(activations, axis=0)
            labels = ops.concat(labels, axis=0)
        return activations, labels

    @staticmethod
    def calculate_activation_statistics(activations):
        activations = activations.numpy()
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def evaluate(self, model, loaders):

        def print_logs(metric, key):
            print(f"Computing action2motion {metric} on the {key} loader ...")

        metrics = {}

        computedfeats = {}
        for key, loader in loaders.items():
            metric = "accuracy"
            print_logs(metric, key)
            mkey = f"{metric}_{key}"
            metrics[mkey], _ = calculate_accuracy(model, loader,  # metrics[mkey] 的内容是 accuracy
                                                  self.num_classes,
                                                  self.gru_classifier, self.device)

            # features for diversity
            print_logs("features", key)
            feats, labels = self.compute_features(model, loader)
            print_logs("stats", key)
            stats = self.calculate_activation_statistics(feats)  # stats = (mu, sigma)

            computedfeats[key] = {"feats": feats,
                                  "labels": labels,
                                  "stats": stats}

            print_logs("diversity", key)
            ret = calculate_diversity_multimodality(feats, labels, self.num_classes)
            metrics[f"diversity_{key}"], metrics[f"multimodality_{key}"] = ret

        # taking the stats of the ground truth and remove it from the computed feats
        gtstats = computedfeats["gt"]["stats"]
        # computing fid
        for key, loader in computedfeats.items():
            metric = "fid"
            mkey = f"{metric}_{key}"

            stats = computedfeats[key]["stats"]
            metrics[mkey] = float(calculate_fid(gtstats, stats))

        return metrics
