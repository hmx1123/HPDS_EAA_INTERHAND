import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from models.encoder import load_encoder
from models.decoder import load_decoder

from utils.config import load_cfg


class Module(nn.Module):
    def __init__(self, encoder, decoder):
        super(Module, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img):
        maps_dict = self.encoder(img)
        result, paramsDict, handDictList, otherInfo = self.decoder(maps_dict)
        # otherInfo.update(maps_dict)

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
    
    # 加载编码器预训练权重
    path = os.path.join(abspath, str(cfg.MODEL_PARAM.ENCODER_PRETRAIN_PATH))
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        print("load model params from {}".format(path))
        
        # 尝试直接加载
        try:
            model.encoder.load_state_dict(state, strict=True)
        except RuntimeError as e:
            # 如果直接加载失败，尝试去除前缀"encoder."（假设有8个字符的前缀）
            print(f"Direct loading failed. Try removing the prefix:")
            state2 = {}
            for k, v in state.items():
                if k.startswith('encoder.'):
                    # 移除"encoder."前缀（8个字符）
                    state2[k[8:]] = v
                else:
                    # 如果没有前缀，保留原键
                    state2[k] = v
            model.encoder.load_state_dict(state2, strict=True)
    else:
        print(f"The encoder pre-trained weight path does not exist: {path}")

    # 加载解码器预训练权重
    path = os.path.join(abspath, str(cfg.MODEL_PARAM.DECODER_PRETRAIN_PATH))
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        print("load model params from {}".format(path))
        
        # 尝试直接加载（使用strict=False允许部分加载）
        try:
            model.decoder.load_state_dict(state, strict=True)
        except RuntimeError as e:
            # 如果直接加载失败，尝试去除前缀
            print(f"Direct loading failed. Try removing the prefix:")
            state2 = {}
            for k, v in state.items():
                if k.startswith('decoder.'):
                    # 移除"decoder."前缀（8个字符）
                    state2[k[8:]] = v
                else:
                    # 如果没有前缀，保留原键
                    state2[k] = v
            model.decoder.load_state_dict(state2, strict=False)
    else:
        print(f"The decoder pre-trained weight path does not exist: {path}")

    return model
