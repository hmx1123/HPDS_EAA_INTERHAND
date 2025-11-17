from models.model import load_model
from ptflops import get_model_complexity_info
import torch
from models.modules import EfficientAdditiveAttention, EAViT
from models.modules import SimpleViT

network = load_model('utils/defaults.yaml')
x = torch.randn(1, 3, 256, 256)
result, paramsDict, handDictList, otherInfo = network(x)
# flops, params = get_model_complexity_info(
#     network, (3, 256, 256), as_strings=True, print_per_layer_stat=True
# )
# print()

# v = EAViT(
#     image_size=64,
#     patch_size=8,
#     num_classes=128,
#     dim=128,
#     depth=6,
#     heads=16,
#     mlp_dim=128,
#     channels=21+3+1,
# )

# img = torch.randn(1, 21+3+1, 64, 64)

# preds = v(img)  # (1, 1000)


# flops, params = get_model_complexity_info(
#     v, (21+3+1, 64, 64), as_strings=True, print_per_layer_stat=False
# )
# print(f"VIT:")
# print(f"FLOPs: {flops}")
# print(f"Params: {params}")
