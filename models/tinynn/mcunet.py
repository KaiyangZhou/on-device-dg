import os
import gdown

import torch
import torch.nn as nn

from dassl.modeling import BACKBONE_REGISTRY, Backbone
from dassl.utils import mkdir_if_missing

from .layers import (
    ConvLayer,
    DsConvLayer,
    InvertedBlock,
    LinearLayer,
    OpSequential,
    ResidualBlock,
)
from .utils import make_divisible

__all__ = ["MCUNet"]


class MCUNet(Backbone):
    def __init__(self, channel_divisor=8, n_classes=1000, dropout_rate=0, ms_class=None, ms_layers=[]):
        super(MCUNet, self).__init__()
        stage_width_list = [16, 8, 16, 24, 40, 48, 96]
        head_width_list = [160]
        act_func = "relu6"

        block_configs = [
            [[3, 5, 5, 4], [7, 3, 7, 5], 4, 2],
            [[5, 5, 5], [5, 5, 5], 3, 2],
            [[5, 6, 4], [3, 7, 5], 3, 2],
            [[5, 5, 5], [5, 7, 3], 3, 1],
            [[6, 5, 4], [3, 7, 3], 3, 2],
        ]

        input_stem = OpSequential(
            [
                ConvLayer(3, stage_width_list[0], 3, 2, act_func=act_func),
                ResidualBlock(
                    DsConvLayer(
                        stage_width_list[0],
                        stage_width_list[1],
                        3,
                        1,
                        (act_func, None),
                    ),
                    shortcut=None,
                ),
            ]
        )

        # stages
        stages = []
        in_channels = stage_width_list[1]
        for (e_list, ks_list, n, s), c in zip(block_configs, stage_width_list[2:]):
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                mid_channels = make_divisible(
                    round(e_list[i] * in_channels), channel_divisor
                )
                mb_conv = ResidualBlock(
                    InvertedBlock(
                        in_channels,
                        c,
                        ks_list[i],
                        stride,
                        mid_channels=mid_channels,
                        act_func=(act_func, act_func, None),
                    ),
                    shortcut=nn.Identity()
                    if (stride == 1 and in_channels == c and i != 0)
                    else None,
                )
                blocks.append(mb_conv)
                in_channels = c
            stages.append(OpSequential(blocks))

        # head
        head = OpSequential(
            [
                ResidualBlock(
                    InvertedBlock(
                        in_channels,
                        head_width_list[0],
                        7,
                        mid_channels=480,
                        act_func=(act_func, act_func, None),
                    ),
                    shortcut=None,
                ),
                nn.AdaptiveAvgPool2d(1),
                # LinearLayer(head_width_list[0], n_classes, dropout_rate=dropout_rate),
            ]
        )

        self.backbone = nn.ModuleDict(
            {
                "input_stem": input_stem,
                "stages": nn.ModuleList(stages),
            }
        )
        self.head = head

        self.mixstyle = None
        self.ms_layers = None
        if ms_layers:
            self.mixstyle = ms_class()
            self.ms_layers = ms_layers
            print(f"Insert {self.mixstyle.__class__.__name__} after layer(s) of {ms_layers}")

        self._out_features = head_width_list[0]

    def forward(self, x: torch.Tensor, output_f_mid: bool = False) -> torch.Tensor:
        f_mid = []
        x = self.backbone["input_stem"](x)
        f_mid.append(x)
        for i, stage in enumerate(self.backbone["stages"]):
            x = stage(x)
            # use i+1 instead of i (count input_stem as the first layer)
            if self.mixstyle is not None and (i + 1) in self.ms_layers:
                x = self.mixstyle(x)
            f_mid.append(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        if output_f_mid:
            return x, f_mid
        else:
            return x


@BACKBONE_REGISTRY.register()
def mcunet(pretrained=True, **kwargs):
    model = MCUNet()
    if pretrained:
        # https://github.com/mit-han-lab/tinyml/tree/master/netaug
        url = "https://drive.google.com/uc?id=1qIH_Lo6-5BwpqAnY7Ymd1QgWOu86Yr2Y"
        model_dir = "~/.torch/oddg_tinynn"
        model_dir = os.path.expanduser(model_dir)
        mkdir_if_missing(model_dir)
        cached_file = os.path.join(model_dir, "mcunet.pth")
        if not os.path.exists(cached_file):
            gdown.download(url, cached_file, quiet=False)
        weights = torch.load(cached_file, map_location="cpu")
        weights = weights["state_dict"]
        model.load_state_dict(weights, strict=False)
    return model


@BACKBONE_REGISTRY.register()
def mcunet_ms_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import MixStyle
    ms_class = MixStyle
    ms_layers = [1, 2]
    model = MCUNet(ms_class=ms_class, ms_layers=ms_layers)
    if pretrained:
        # https://github.com/mit-han-lab/tinyml/tree/master/netaug
        url = "https://drive.google.com/uc?id=1qIH_Lo6-5BwpqAnY7Ymd1QgWOu86Yr2Y"
        model_dir = "~/.torch/oddg_tinynn"
        model_dir = os.path.expanduser(model_dir)
        mkdir_if_missing(model_dir)
        cached_file = os.path.join(model_dir, "mcunet.pth")
        if not os.path.exists(cached_file):
            gdown.download(url, cached_file, quiet=False)
        weights = torch.load(cached_file, map_location="cpu")
        weights = weights["state_dict"]
        model.load_state_dict(weights, strict=False)
    return model


@BACKBONE_REGISTRY.register()
def mcunet_efdmix_l12(pretrained=True, **kwargs):
    from dassl.modeling.ops import EFDMix
    ms_class = EFDMix
    ms_layers = [1, 2]
    model = MCUNet(ms_class=ms_class, ms_layers=ms_layers)
    if pretrained:
        # https://github.com/mit-han-lab/tinyml/tree/master/netaug
        url = "https://drive.google.com/uc?id=1qIH_Lo6-5BwpqAnY7Ymd1QgWOu86Yr2Y"
        model_dir = "~/.torch/oddg_tinynn"
        model_dir = os.path.expanduser(model_dir)
        mkdir_if_missing(model_dir)
        cached_file = os.path.join(model_dir, "mcunet.pth")
        if not os.path.exists(cached_file):
            gdown.download(url, cached_file, quiet=False)
        weights = torch.load(cached_file, map_location="cpu")
        weights = weights["state_dict"]
        model.load_state_dict(weights, strict=False)
    return model
