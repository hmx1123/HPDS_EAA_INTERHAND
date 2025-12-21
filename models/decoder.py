from models.modules.InvertedResidual import (
    InvertedResidual,
    DepthWiseSeparable,
    DepthWiseSeparableRes,
)
from models.modules import GCN_vert_convert, DualGraph, EAViT
from utils.utils import (
    projection_batch,
    get_dense_color_path,
    get_graph_dict_path,
    get_upsample_path,
    get_mesh_dict_path,
)
from dataset.dataset_utils import IMG_SIZE, BONE_LENGTH
import numpy as np
import pickle
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class decoder(nn.Module):
    def __init__(
        self,
        map_dim=50,
        vit_dim=256,
        gcn_in_dim=[256, 128, 128],
        gcn_out_dim=[128, 128, 64],
        graph_k=2,
        graph_layer_num=4,
        left_graph_dict={},
        right_graph_dict={},
        left_mesh_dict={},
        right_mesh_dict={},
        vertex_num=778,
        dense_coor=None,
        num_attn_heads=4,
        upsample_weight=None,
        dropout=0.05,
    ):
        super(decoder, self).__init__()
        for i in range(len(gcn_out_dim) - 1):
            assert gcn_out_dim[i] == gcn_in_dim[i + 1]

        graph_dict = {"left": left_graph_dict, "right": right_graph_dict}
        graph_dict["left"]["coarsen_graphs_L"].reverse()
        graph_dict["right"]["coarsen_graphs_L"].reverse()
        mesh_dict = {"left": left_mesh_dict, "right": right_mesh_dict}
        graph_L = {}
        mesh_L = {}
        for hand_type in ["left", "right"]:
            graph_L[hand_type] = graph_dict[hand_type]["coarsen_graphs_L"]
            mesh_L[hand_type] = mesh_dict[hand_type]["mesh_L"]

        self.vNum_in = graph_L["left"][0].shape[0]
        self.vNum_out = graph_L["left"][2].shape[0]
        self.vNum_all = graph_L["left"][-1].shape[0]
        self.vNum_mano = vertex_num
        self.gcn_in_dim = gcn_in_dim
        self.gcn_out_dim = gcn_out_dim

        if dense_coor is not None:
            dense_coor = torch.from_numpy(dense_coor).float()
            self.register_buffer("dense_coor", dense_coor)

        self.converter = {}
        for hand_type in ["left", "right"]:
            self.converter[hand_type] = GCN_vert_convert(
                vertex_num=self.vNum_mano,
                graph_perm_reverse=graph_dict[hand_type]["graph_perm_reverse"],
                graph_perm=graph_dict[hand_type]["graph_perm"],
            )

        self.vit = EAViT(
            image_size=64,
            patch_size=8,
            num_classes=self.gcn_in_dim[0],
            dim=vit_dim,
            depth=6,
            heads=vit_dim // 64,
            mlp_dim=vit_dim // 2,
            channels=map_dim,
        )

        self.dual_gcn = DualGraph(
            verts_in_dim=self.gcn_in_dim,
            verts_out_dim=self.gcn_out_dim,
            graph_L_Left=graph_L["left"][:3] + [mesh_L["left"]],
            graph_L_Right=graph_L["right"][:3] + [mesh_L["right"]],
            graph_k=[graph_k, graph_k, graph_k, graph_k],
            graph_layer_num=[
                graph_layer_num,
                graph_layer_num,
                graph_layer_num,
                graph_layer_num,
            ],
            dropout=dropout,
        )

        self.unsample_layer = nn.Linear(self.vNum_out, self.vNum_mano, bias=False)

        self.avg_head = nn.Linear(self.gcn_in_dim[0], 3)
        self.coord_avg_head = nn.AvgPool1d(
            kernel_size=self.gcn_out_dim[-1] // 3, stride=self.gcn_out_dim[-1] // 3
        )
        self.coord_head = nn.Linear(self.gcn_out_dim[-1], 3)

        self.graph_upsample = nn.Upsample(size=1008)

        for m in self.modules():
            weights_init(m)

        if upsample_weight is not None:
            state = {
                "weight": upsample_weight.to(self.unsample_layer.weight.data.device)
            }
            self.unsample_layer.load_state_dict(state)
        else:
            weights_init(self.unsample_layer)

    def get_upsample_weight(self):
        return self.unsample_layer.weight.data

    def forward(self, Map):
        map = torch.cat([v for _, v in Map.items()], dim=1)

        grid_fmaps = self.vit(map)

        Lf = grid_fmaps[:, : self.vNum_in]
        Rf = grid_fmaps[:, self.vNum_in + 1 : -1]
        Lf, Rf = self.dual_gcn(Lf, Rf)

        scale = {}
        trans2d = {}
        temp = grid_fmaps[:, self.vNum_in].squeeze(dim=1)
        temp = self.avg_head(temp)
        scale["left"] = temp[:, 0]
        trans2d["left"] = temp[:, 1:]
        temp = grid_fmaps[:, -1].squeeze(dim=1)
        temp = self.avg_head(temp)
        scale["right"] = temp[:, 0]
        trans2d["right"] = temp[:, 1:]
        paramsDict = {"scale": scale, "trans2d": trans2d}

        handDictList = []
        verts3d = {"left": self.coord_head(Lf), "right": self.coord_head(Rf)}
        verts2d = {}
        result = {"verts3d": {}, "verts2d": {}}
        for hand_type in ["left", "right"]:
            verts2d[hand_type] = projection_batch(
                scale[hand_type],
                trans2d[hand_type],
                verts3d[hand_type],
                img_size=IMG_SIZE,
            )
            # result["verts3d"][hand_type] = self.unsample_layer(
            #     verts3d[hand_type].transpose(1, 2)
            # ).transpose(1, 2)
            result["verts3d"][hand_type] = verts3d[hand_type]
            result["verts2d"][hand_type] = projection_batch(
                scale[hand_type],
                trans2d[hand_type],
                result["verts3d"][hand_type],
                img_size=IMG_SIZE,
            )
        handDictList.append({"verts3d": verts3d, "verts2d": verts2d})

        otherInfo = {}
        otherInfo["verts3d_MANO_list"] = {"left": [], "right": []}
        otherInfo["verts2d_MANO_list"] = {"left": [], "right": []}
        for i in range(len(handDictList)):
            for hand_type in ["left", "right"]:
                v = handDictList[i]["verts3d"][hand_type]
                v = v.permute(0, 2, 1).contiguous()
                v = self.graph_upsample(v)
                v = v.permute(0, 2, 1).contiguous()
                otherInfo["verts3d_MANO_list"][hand_type].append(
                    self.converter[hand_type].GCN_to_vert(v)
                )
                v = handDictList[i]["verts2d"][hand_type]
                v = v.permute(0, 2, 1).contiguous()
                v = self.graph_upsample(v)
                v = v.permute(0, 2, 1).contiguous()
                otherInfo["verts2d_MANO_list"][hand_type].append(
                    self.converter[hand_type].GCN_to_vert(v)
                )
        otherInfo.update(Map)

        return result, paramsDict, handDictList, otherInfo


def load_decoder(cfg):
    graph_path = get_graph_dict_path()
    with open(graph_path["left"], "rb") as file:
        left_graph_dict = pickle.load(file)
    with open(graph_path["right"], "rb") as file:
        right_graph_dict = pickle.load(file)

    dense_path = get_dense_color_path()
    with open(dense_path, "rb") as file:
        dense_coor = pickle.load(file)

    upsample_path = get_upsample_path()
    with open(upsample_path, "rb") as file:
        upsample_weight = pickle.load(file)
    upsample_weight = torch.from_numpy(upsample_weight).float()

    mesh_dict_path = get_mesh_dict_path()
    with open(mesh_dict_path["left"], "rb") as file:
        left_mesh_dict = pickle.load(file)
    with open(mesh_dict_path["right"], "rb") as file:
        right_mesh_dict = pickle.load(file)

    # directory='./misc/graphs_adj_128x128.pkl'
    # with open(directory, "rb") as file:
    #     graphs_adj_128x128 = pickle.load(file)

    model = decoder(
        map_dim=cfg.MODEL.HRNet_MODEL.NUM_CLASSES,
        vit_dim=cfg.MODEL.VIT_DIM,
        gcn_in_dim=cfg.MODEL.GCN_IN_DIM,
        gcn_out_dim=cfg.MODEL.GCN_OUT_DIM,
        graph_k=cfg.MODEL.graph_k,
        graph_layer_num=cfg.MODEL.graph_layer_num,
        vertex_num=778,
        dense_coor=dense_coor,
        left_graph_dict=left_graph_dict,
        right_graph_dict=right_graph_dict,
        left_mesh_dict=left_mesh_dict,
        right_mesh_dict=right_mesh_dict,
        num_attn_heads=16,
        upsample_weight=upsample_weight,
        dropout=cfg.TRAIN.dropout,
    )

    return model
