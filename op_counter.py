import argparse
from tabulate import tabulate

import torch
import torch.nn as nn

from thop import profile, clever_format

from dassl.config import get_cfg_default
from dassl.engine.trainer import SimpleNet

import models.mobilenetv3
import models.tinynn.tiny_mbv2
import models.tinynn.mcunet

def estimate_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275/2
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="mcunet")
parser.add_argument("-r", "--resolution", type=int, default=224)
parser.add_argument("-n", "--num_classes", type=int, default=1000)
args = parser.parse_args()

cfg = get_cfg_default()
cfg.MODEL.BACKBONE.NAME = args.model
model = SimpleNet(cfg, cfg.MODEL, args.num_classes)
print(f"Classes: {args.num_classes:,}")

imsize = args.resolution
print(f"Input size: {imsize}x{imsize}")

input = torch.randn(1, 3, imsize, imsize)
macs, params = profile(model, inputs=(input, ), verbose=False)
macs, params = clever_format([macs, params], "%.2f")
model_size = estimate_size(model)
model_size = f"{model_size:.2f}MB"

table = [
    ["Params", params],
    ["Size", model_size],
    ["MACs", macs]
]

print(tabulate(table))
