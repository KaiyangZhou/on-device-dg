"""
Credit: https://github.com/HobbitLong/RepDistiller
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.utils import count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from .kd import KD


class ConvReg(nn.Module):
    """Convolutional regression for FitNet."""
    
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplementedError("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


@TRAINER_REGISTRY.register()
class FitNet(KD):
    """
    FitNets: Hints for Thin Deep Nets.
    """

    hint_layer = 2  # index of hint layer {0, 1, 2, ...}
    beta = 1.0

    def __init__(self, cfg):
        self.hparam_table.append(["hint_layer", self.hint_layer])
        super().__init__(cfg)
        self.build_regressor()
    
    def build_regressor(self):
        cfg = self.cfg

        input = torch.rand(2, 3, *cfg.INPUT.SIZE)
        input = input.to(self.device)
        with torch.no_grad():
            _, feature_model = self.model(input, output_f_mid=True)
            _, feature_teacher = self.teacher(input, output_f_mid=True)
            f_m = feature_model[self.hint_layer]
            f_t = feature_teacher[self.hint_layer]
        # print(f_m.shape, f_t.shape)

        print("Building regressor")
        self.regressor = ConvReg(f_m.shape, f_t.shape)
        self.regressor.to(self.device)
        print(f"# params: {count_num_param(self.regressor):,}")
        self.optim_r = build_optimizer(self.regressor, cfg.OPTIM)
        self.sched_r = build_lr_scheduler(self.optim_r, cfg.OPTIM)
        self.register_model("regressor", self.regressor, self.optim_r, self.sched_r)
    
    def compute_kd(self, feature_model, feature_teacher, batch):
        f_m = self.regressor(feature_model[self.hint_layer])
        f_t = feature_teacher[self.hint_layer]
        return F.mse_loss(f_m, f_t)
