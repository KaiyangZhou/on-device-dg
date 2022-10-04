"""
Credit: https://github.com/HobbitLong/RepDistiller
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param

from .kd import KD


class VIDLoss(nn.Module):
    """Variational Information Distillation Loss."""
    
    def __init__(
        self,
        num_input_channels,
        num_mid_channel,
        num_target_channels,
        init_pred_var=5.0,
        eps=1e-5):
        super().__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var + torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss


@TRAINER_REGISTRY.register()
class VID(KD):
    """
    Variational Information Distillation for Knowledge Transfer.
    """

    beta = 1.0

    def __init__(self, cfg):
        super().__init__(cfg)
        self.build_vid()
    
    def build_vid(self):
        cfg = self.cfg
        
        input = torch.rand(2, 3, *cfg.INPUT.SIZE)
        input = input.to(self.device)
        with torch.no_grad():
            _, feature_model = self.model(input, output_f_mid=True)
            _, feature_teacher = self.teacher(input, output_f_mid=True)
        
        m_n = [f.shape[1] for f in feature_model[1:-1]]
        t_n = [f.shape[1] for f in feature_teacher[1:-1]]

        print("Building vid-learner")
        self.vid_learner = nn.ModuleList([VIDLoss(m, t, t) for m, t in zip(m_n, t_n)])
        self.vid_learner.to(self.device)
        print(f"# params: {count_num_param(self.vid_learner):,}")
        self.optim_vid = build_optimizer(self.vid_learner, cfg.OPTIM)
        self.sched_vid = build_lr_scheduler(self.optim_vid, cfg.OPTIM)
        self.register_model("vid_learner", self.vid_learner, self.optim_vid, self.sched_vid)
    
    def compute_kd(self, feature_model, feature_teacher, batch):
        g_m = feature_model[1:-1]
        g_t = feature_teacher[1:-1]
        losses = [vid(f_m, f_t) for f_m, f_t, vid in zip(g_m, g_t, self.vid_learner)]
        return sum(losses)
