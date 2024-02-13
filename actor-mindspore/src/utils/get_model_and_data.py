# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: get_model_and_data.py
@Date: 2023/4/30 9:50
@Author: caijianfeng
"""
from ..datasets.get_dataset import get_datasets
from ..recognition.get_model import get_model as get_rec_model
from ..models.get_model import get_model as get_gen_model


def get_model_and_data(parameters):
    # datasets is a dict: {'train': train_dataset, 'test': test_dataset}
    datasets = get_datasets(parameters)

    if parameters["modelname"] == "recognition":
        model = get_rec_model(parameters)
    else:
        model = get_gen_model(parameters)
    return model, datasets
