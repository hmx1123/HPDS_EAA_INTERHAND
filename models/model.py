import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from dataset.dataset_utils import IMG_SIZE
from models.encoder import load_encoder
from models.decoder import load_decoder

from utils.config import load_cfg


class Module(nn.Module):
    def __init__(self, encoder, decoder):
        super(Module, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img):
        hms, mask, dp = self.encoder(img)
        result, paramsDict, handDictList, otherInfo = self.decoder(hms, mask, dp)

        if hms is not None:
            otherInfo["hms"] = hms
        if mask is not None:
            otherInfo["mask"] = mask
        if dp is not None:
            otherInfo["dense"] = dp

        return result, paramsDict, handDictList, otherInfo


def load_model(cfg):
    if isinstance(cfg, str):
        cfg = load_cfg(cfg)
    encoder = load_encoder(cfg)
    decoder = load_decoder(cfg)
    model = Module(encoder, decoder)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(
        model, (3, 256, 256), as_strings=True, print_per_layer_stat=False
    )
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(abspath, str(cfg.MODEL_PARAM.MODEL_PRETRAIN_PATH))
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        print("load model params from {}".format(path))
        try:
            model.load_state_dict(state)
        except:
            state2 = {}
            for k, v in state.items():
                state2[k[7:]] = v
            model.load_state_dict(state2, strict=False)

    return model
