# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: get_model.py
@Date: 2023/4/27 10:23
@Author: caijianfeng
"""
import importlib

JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

LOSSES = ["rc", "kl", "rcxyz"]  # not used: "hp", "mmd", "vel", "velxyz"

MODELTYPES = ["cvae"]  # not used: "cae"
ARCHINAMES = ["fc", "gru", "transformer", "transgru", "grutrans", "autotrans"]


def get_model(parameters):
    modeltype = parameters["modeltype"]
    archiname = parameters["archiname"]

    archi_module = importlib.import_module(f'.architectures.{archiname}', package="src.models")
    Encoder = archi_module.__getattribute__(f"Encoder_{archiname.upper()}")
    Decoder = archi_module.__getattribute__(f"Decoder_{archiname.upper()}")

    model_module = importlib.import_module(f'.modeltype.{modeltype}', package="src.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")

    encoder = Encoder(**parameters)
    decoder = Decoder(**parameters)

    parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]  # parameters["outputxyz"] = True
    return Model(encoder, decoder, **parameters)
