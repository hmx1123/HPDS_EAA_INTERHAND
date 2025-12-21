from models.modules.EfficientAdditiveAttention import EfficientAdditiveAttention
import torchvision.models as models
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from utils.config import load_cfg
from models.modules.InvertedResidual import DepthWiseSeparable, DepthWiseSeparableRes
from models.modules import (
    weights_init,
)
from models.manolayer import ManoLayer
from utils.utils import projection_batch
from dataset.dataset_utils import IMG_SIZE
import numpy as np
import pickle
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ResNetSimple_decoder(nn.Module):
    def __init__(
        self,
        expansion=4,
        in_fDim=[256, 256, 256, 256],
        out_fDim=[128, 128, 128, 128],
        direction=["flat", "up", "up", "up"],
        out_dim=3,
        e=[0.25, 0.25, 0.25, 0.25],
        hid_layer=[2, 3, 4, 5],
    ):
        super(ResNetSimple_decoder, self).__init__()
        self.models = nn.ModuleList()
        for i in range(len(direction)):
            self.models.append(
                self.make_layer(
                    in_fDim[i],
                    out_fDim[i],
                    direction[i],
                    kernel_size=3,
                    hid_layer=hid_layer[i],
                    padding=1,
                    e=e[i],
                )
            )

        self.final_layer = nn.Sequential(
            nn.Conv2d(out_fDim[-1], out_dim, 1), nn.BatchNorm2d(out_dim)
        )

    def make_layer(
        self, in_dim, out_dim, direction, kernel_size=3, hid_layer=2, padding=1, e=0.25
    ):
        assert direction in ["flat", "up"]

        layers = []
        if direction == "up":
            layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            )
        layers.append(
            DepthWiseSeparableRes(
                in_dim, out_dim, hid_layer=hid_layer, kernel=kernel_size, e=e
            )
        )
        # layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        # layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.BatchNorm2d(out_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        in_x = x[0]
        fmaps = []
        for i in range(len(self.models)):
            # HRnet连接桥
            if i != 0:
                in_x = torch.cat((in_x, x[i - 1]), dim=1)
            #
            in_x = self.models[i](in_x)
            fmaps.append(in_x)
        output = self.final_layer(in_x)
        return output, fmaps


class ResNetSimple(nn.Module):
    def __init__(
        self,
        model_type="resnet50",
        pretrained=False,
        in_fmapDim=[256, 256, 256, 256],
        out_fmapDim=[128, 128, 128, 128],
        handNum=2,
        heatmapDim=21,
    ):
        super(ResNetSimple, self).__init__()
        assert model_type in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]
        if model_type == "resnet18":
            self.resnet = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.expansion = 1
        elif model_type == "resnet34":
            self.resnet = resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.expansion = 1
        elif model_type == "resnet50":
            self.resnet = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.expansion = 4
        elif model_type == "resnet101":
            self.resnet = resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            self.expansion = 4
        elif model_type == "resnet152":
            self.resnet = resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            self.expansion = 4

        # self.hms_decoder = ResNetSimple_decoder(
        #     expansion=self.expansion,
        #     fDim=fmapDim,
        #     direction=["flat", "up", "up", "up"],
        #     out_dim=heatmapDim * handNum,
        #     # e=[0.84, 0.84, 0.84, 0.84],
        # )

        # self.dp_decoder = ResNetSimple_decoder(
        #     expansion=self.expansion,
        #     fDim=fmapDim,
        #     direction=["flat", "up", "up", "up"],
        #     out_dim=3 * handNum,
        #     # e=[0.12, 0.12, 0.12, 0.12],
        # )

        # self.mask_decoder = ResNetSimple_decoder(
        #     expansion=self.expansion,
        #     fDim=fmapDim,
        #     direction=["flat", "up", "up", "up"],
        #     out_dim=handNum,
        #     # e=[0.04, 0.04, 0.04, 0.04],
        # )

        self.f_maps_decoder = ResNetSimple_decoder(
            expansion=self.expansion,
            in_fDim=in_fmapDim,
            out_fDim=out_fmapDim,
            direction=["flat", "up", "up", "up"],
            out_dim=heatmapDim * handNum + 3 * handNum + handNum,
            e=[0.75, 0.75, 0.75, 0.75],
        )
        self.handNum = handNum
        self.heatmapDim = heatmapDim

        for m in self.modules():
            weights_init(m)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x4 = self.resnet.layer1(x)
        x3 = self.resnet.layer2(x4)
        x2 = self.resnet.layer3(x3)
        x1 = self.resnet.layer4(x2)

        img_fmaps = [x1, x2, x3, x4]

        # hms, hms_fmaps = self.hms_decoder(x1)
        # dp, dp_fmaps = self.dp_decoder(x1)
        # mask, mask_fmaps = self.mask_decoder(x1)
        fmaps, grid_fmaps = self.f_maps_decoder(img_fmaps)
        hms = fmaps[:, : self.heatmapDim * self.handNum]
        # hms=gaussian_nms(hms)
        dp = fmaps[
            :,
            self.heatmapDim * self.handNum : self.heatmapDim * self.handNum
            + 3 * self.handNum,
        ]
        mask = fmaps[
            :,
            self.heatmapDim * self.handNum
            + 3 * self.handNum : self.heatmapDim * self.handNum
            + 3 * self.handNum
            + self.handNum,
        ]

        return hms, mask, dp


class HRNet(nn.Module):
    def __init__(
        self,
        cfg=None,
        handNum=2,
    ):
        super(HRNet, self).__init__()

        self.HRNet_OUTPUT = cfg.HRNet_OUTPUT
        self.handNum = handNum
        
        cfg.NUM_CLASSES = sum(
            [
                21 * self.handNum if "hms" in self.HRNet_OUTPUT else 0,
                3 * self.handNum if "dense" in self.HRNet_OUTPUT else 0,
                self.handNum if "mask" in self.HRNet_OUTPUT else 0,
            ]
        )

        from models.modules.hrnet import get_model

        self.model = get_model(cfg)

    def forward(self, x):
        fmaps = self.model(x)

        idx = 0
        maps_dict={}

        if "hms" in self.HRNet_OUTPUT:
            hms = fmaps[:, idx : (idx := idx + 21 * self.handNum)]
            maps_dict['hms']=hms
        if "mask" in self.HRNet_OUTPUT:
            mask = fmaps[:, idx : (idx := idx + self.handNum)]
            maps_dict['mask']=mask
        if "dense" in self.HRNet_OUTPUT:
            dp = fmaps[:, idx : (idx := idx + 3 * self.handNum)]
            maps_dict['dense']=dp

        return maps_dict


def load_encoder(cfg):
    if cfg.MODEL.ENCODER_TYPE.find("resnet") != -1:
        encoder = ResNetSimple(
            model_type=cfg.MODEL.ENCODER_TYPE,
            pretrained=True,
            # HRnet:[512, 128+512, 128+256, 128+128], ResNet:[512, 128, 128, 128]
            in_fmapDim=[512, 128 + 512, 128 + 256, 128 + 128],
            #
            out_fmapDim=[128, 128, 128, 128],
            handNum=2,
            heatmapDim=21,
        )
    elif cfg.MODEL.ENCODER_TYPE.find("hrnet") != -1:
        from models.modules.hrnet import get_model

        encoder = HRNet(
            cfg=cfg.MODEL.HRNet_MODEL,
            handNum=2,
        )

    return encoder
