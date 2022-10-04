import torch
import torch.nn as nn

from dassl.modeling import Backbone

from .layers import (
    ConvLayer,
    DsConvLayer,
    InvertedBlock,
    LinearLayer,
    OpSequential,
    ResidualBlock,
)

from .utils import make_divisible

__all__ = ["MobileNetV2"]


class MobileNetV2(Backbone):
    def __init__(
        self, width_mult=1.0, channel_divisor=8, n_classes=1000, dropout_rate=0,
        ms_class=None, ms_layers=[]
    ):
        super(MobileNetV2, self).__init__()
        stage_width_list = [32, 16, 24, 32, 64, 96, 160]
        head_width_list = [320, 1280]
        act_func = "relu6"

        block_configs = [
            # t, n, s
            [6, 2, 2],
            [6, 3, 2],
            [6, 4, 2],
            [6, 3, 1],
            [6, 3, 2],
        ]

        for i, w in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(w * width_mult, channel_divisor)
        for i, w in enumerate(head_width_list):
            head_width_list[i] = make_divisible(w * width_mult, channel_divisor)
        head_width_list[1] = max(head_width_list[1], 1280)

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
        for (t, n, s), c in zip(block_configs, stage_width_list[2:]):
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                mid_channels = make_divisible(round(t * in_channels), channel_divisor)
                mb_conv = ResidualBlock(
                    InvertedBlock(
                        in_channels,
                        c,
                        3,
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
                        3,
                        expand_ratio=6,
                        act_func=(act_func, act_func, None),
                    ),
                    shortcut=None,
                ),
                ConvLayer(head_width_list[0], head_width_list[1], 1, act_func=act_func),
                nn.AdaptiveAvgPool2d(1),
                # LinearLayer(head_width_list[1], n_classes, dropout_rate=dropout_rate),
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

        self._out_features = head_width_list[1]

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
