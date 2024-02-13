# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: get_model.py
@Date: 2023/4/30 9:51
@Author: caijianfeng
"""
from .models.stgcn import STGCN


def get_model(parameters):
    layout = "smpl" if parameters["glob"] else "smpl_noglobal"

    model = STGCN(in_channels=parameters["nfeats"],
                  num_class=parameters["num_classes"],
                  graph_args={"layout": layout, "strategy": "spatial"},
                  edge_importance_weighting=True)

    # model = model.to(parameters["device"])
    return model

