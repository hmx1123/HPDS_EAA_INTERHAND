from .DualGraph import DualGraph
from .EfficientAdditiveAttention import EfficientAdditiveAttention, EAViT
from .coarsening import build_graph
import torch.nn as nn


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class GCN_vert_convert():
    def __init__(self, vertex_num=1, graph_perm_reverse=[0], graph_perm=[0]):
        self.graph_perm_reverse = graph_perm_reverse[:vertex_num]
        self.graph_perm = graph_perm

    def vert_to_GCN(self, x):
        # x: B x v x f
        return x[:, self.graph_perm]

    def GCN_to_vert(self, x):
        # x: B x v x f
        return x[:, self.graph_perm_reverse]
