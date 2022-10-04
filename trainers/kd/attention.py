"""
Credit: https://github.com/HobbitLong/RepDistiller
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY

from .kd import KD


class ATLoss(nn.Module):
    """Attention Transfer Loss."""

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


@TRAINER_REGISTRY.register()
class Attention(KD):
    """
    Paying More Attention to Attention: Improving the Performance
    of Convolutional Neural Networks via Attention Transfer.
    """

    beta = 1.0

    def __init__(self, cfg):
        super().__init__(cfg)
        self.kd_loss = ATLoss()
    
    def compute_kd(self, feature_model, feature_teacher, batch):
        f_m = feature_model[1:-1]
        f_t = feature_teacher[1:-1]
        return sum(self.kd_loss(f_m, f_t))
