import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torch import Tensor
from .gcn import GraphLayer
from .EfficientAdditiveAttention import EfficientAdditiveAttention, FeedForward


def graph_upsample(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
        x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
        return x
    else:
        return x


class DualGraphLayer(nn.Module):
    def __init__(
        self,
        verts_in_dim=256,
        verts_out_dim=256,
        graph_L_Left=None,
        graph_L_Right=None,
        graph_k=2,
        graph_layer_num=4,
        dropout=0.01,
    ):
        super().__init__()
        self.verts_num = graph_L_Left.shape[0]
        self.verts_in_dim = verts_in_dim
        self.verts_out_dim = verts_out_dim

        self.graph_left = GraphLayer(
            verts_in_dim,
            verts_out_dim,
            graph_L_Left,
            graph_k,
            graph_layer_num,
            dropout,
        )
        self.graph_right = GraphLayer(
            verts_in_dim,
            verts_out_dim,
            graph_L_Right,
            graph_k,
            graph_layer_num,
            dropout,
        )
        self.inter_atten = nn.Sequential(
                EfficientAdditiveAttention(
                    in_dims=verts_out_dim, num_heads=4, dropout=dropout
                ),
                FeedForward(verts_out_dim, verts_out_dim),
        )

        self.pos_emb_l = nn.Parameter(
            torch.zeros(1, self.verts_num, self.verts_out_dim)
        )
        self.pos_emb_r = nn.Parameter(
            torch.zeros(1, self.verts_num, self.verts_out_dim)
        )

    def forward(self, Lf, Rf):
        BS1, V, f = Lf.shape
        assert V == self.verts_num
        assert f == self.verts_in_dim
        BS2, V, f = Rf.shape
        assert V == self.verts_num
        assert f == self.verts_in_dim

        Lf = self.graph_left(Lf)
        Rf = self.graph_right(Rf)
        
        feat=self.inter_atten(torch.cat((Lf,Rf),dim=1))
        Lf = feat[:, :V]
        Rf = feat[:, V:]

        return Lf, Rf


class DualGraph(nn.Module):
    def __init__(
        self,
        verts_in_dim=[512, 256, 128],
        verts_out_dim=[256, 128, 64],
        graph_L_Left=None,
        graph_L_Right=None,
        graph_k=[2, 2, 2],
        graph_layer_num=[4, 4, 4],
        dropout=0.01,
    ):
        super().__init__()
        self.verts_in_dim = verts_in_dim
        for i in range(len(verts_in_dim) - 1):
            assert verts_out_dim[i] == verts_in_dim[i + 1]
        # for i in range(len(verts_in_dim) - 1):
        #     assert graph_L_Left[i + 1].shape[0] == 2 * graph_L_Left[i].shape[0]
        #     assert graph_L_Right[i + 1].shape[0] == 2 * graph_L_Right[i].shape[0]

        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(len(verts_in_dim)):
            self.layers.append(
                DualGraphLayer(
                    verts_in_dim=verts_in_dim[i],
                    verts_out_dim=verts_out_dim[i],
                    graph_L_Left=graph_L_Left[i],
                    graph_L_Right=graph_L_Right[i],
                    graph_k=graph_k[i],
                    graph_layer_num=graph_layer_num[i],
                    dropout=dropout,
                )
            )
            if i != len(verts_in_dim) - 1:
                if i == 2:
                    self.upsample.append(nn.Upsample(size=778))
                else:
                    self.upsample.append(nn.Upsample(scale_factor=2))

    def forward(
        self,
        Lf,
        Rf,
    ):
        for i in range(len(self.layers)):
            Lf, Rf = self.layers[i](Lf, Rf)

            if i != len(self.layers) - 1:
                Lf = Lf.permute(0, 2, 1).contiguous()  # x = B x F x V
                Lf = self.upsample[i](Lf)  # B x F x (V*p)
                Lf = Lf.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F

                Rf = Rf.permute(0, 2, 1).contiguous()  # x = B x F x V
                Rf = self.upsample[i](Rf)  # B x F x (V*p)
                Rf = Rf.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F

        return Lf, Rf
