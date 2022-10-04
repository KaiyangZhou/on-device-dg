import os
import gdown

import torch
import torch.nn as nn

from dassl.modeling import BACKBONE_REGISTRY
from dassl.utils import mkdir_if_missing

from .layers import (
    ConvLayer,
    InvertedBlock,
    LinearLayer,
    OpSequential,
    ResidualBlock,
)
from .mbv2 import MobileNetV2


class TinyMobileNetV2(MobileNetV2):
    def __init__(self, channel_divisor=8, n_classes=1000, dropout_rate=0, ms_class=None, ms_layers=[]):
        super(TinyMobileNetV2, self).__init__(
            0.35, channel_divisor, n_classes, dropout_rate, ms_class, ms_layers
        )

        self.head = OpSequential(
            [
                ResidualBlock(
                    InvertedBlock(
                        56,
                        112,
                        3,
                        expand_ratio=6,
                        act_func=("relu6", "relu6", None),
                    ),
                    shortcut=None,
                ),
                ConvLayer(112, 448, 1, act_func="relu6"),
                nn.AdaptiveAvgPool2d(1),
                # LinearLayer(448, n_classes, dropout_rate=dropout_rate),
            ]
        )

        self._out_features = 448


@BACKBONE_REGISTRY.register()
def mobilenet_v2_tiny(pretrained=True, **kwargs):
    model = TinyMobileNetV2()
    if pretrained:
        # https://github.com/mit-han-lab/tinyml/tree/master/netaug
        url = "https://drive.google.com/uc?id=14snkQ_CNCkPF5Iu58wlTDNt_xF3m0b0Q"
        model_dir = "~/.torch/oddg_tinynn"
        model_dir = os.path.expanduser(model_dir)
        mkdir_if_missing(model_dir)
        cached_file = os.path.join(model_dir, "mobilenet_v2_tiny.pth")
        if not os.path.exists(cached_file):
            gdown.download(url, cached_file, quiet=False)
        weights = torch.load(cached_file, map_location="cpu")
        model.load_state_dict(weights, strict=False)
    return model


@BACKBONE_REGISTRY.register()
def mobilenet_v2_tiny_ms_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle
    ms_class = MixStyle
    ms_layers = [1, 2]
    model = TinyMobileNetV2(ms_class=ms_class, ms_layers=ms_layers)
    if pretrained:
        # https://github.com/mit-han-lab/tinyml/tree/master/netaug
        url = "https://drive.google.com/uc?id=14snkQ_CNCkPF5Iu58wlTDNt_xF3m0b0Q"
        model_dir = "~/.torch/oddg_tinynn"
        model_dir = os.path.expanduser(model_dir)
        mkdir_if_missing(model_dir)
        cached_file = os.path.join(model_dir, "mobilenet_v2_tiny.pth")
        if not os.path.exists(cached_file):
            gdown.download(url, cached_file, quiet=False)
        weights = torch.load(cached_file, map_location="cpu")
        model.load_state_dict(weights, strict=False)
    return model


@BACKBONE_REGISTRY.register()
def mobilenet_v2_tiny_efdmix_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix
    ms_class = EFDMix
    ms_layers = [1, 2]
    model = TinyMobileNetV2(ms_class=ms_class, ms_layers=ms_layers)
    if pretrained:
        # https://github.com/mit-han-lab/tinyml/tree/master/netaug
        url = "https://drive.google.com/uc?id=14snkQ_CNCkPF5Iu58wlTDNt_xF3m0b0Q"
        model_dir = "~/.torch/oddg_tinynn"
        model_dir = os.path.expanduser(model_dir)
        mkdir_if_missing(model_dir)
        cached_file = os.path.join(model_dir, "mobilenet_v2_tiny.pth")
        if not os.path.exists(cached_file):
            gdown.download(url, cached_file, quiet=False)
        weights = torch.load(cached_file, map_location="cpu")
        model.load_state_dict(weights, strict=False)
    return model
