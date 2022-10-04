"""
Credit: https://github.com/HobbitLong/RepDistiller
"""
import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY

from .kd import KD


class SPLoss(nn.Module):
    """Similarity-Preserving Loss."""
    
    def __init__(self):
        super().__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        # loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        loss = (G_diff * G_diff).sum() / (bsz * bsz)
        return loss


@TRAINER_REGISTRY.register()
class Similarity(KD):
    """
    Similarity-Preserving Knowledge Distillation.
    """

    beta = 0.1

    def __init__(self, cfg):
        super().__init__(cfg)
        self.kd_loss = SPLoss()
    
    def compute_kd(self, feature_model, feature_teacher, batch):
        g_m = [feature_model[-2]]
        g_t = [feature_teacher[-2]]
        return sum(self.kd_loss(g_m, g_t))
