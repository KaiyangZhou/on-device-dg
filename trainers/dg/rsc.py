"""
Credit: https://github.com/facebookresearch/DomainBed
"""
import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy

import torch.autograd as autograd
import numpy as np


@TRAINER_REGISTRY.register()
class RSC(TrainerX):
    """
    https://arxiv.org/abs/2007.02454
    """
    
    f_drop_factor = 0.1
    b_drop_factor = 0.1

    def __init__(self, cfg):
        super().__init__(cfg)
        self.drop_f = (1 - self.f_drop_factor) * 100
        self.drop_b = (1 - self.b_drop_factor) * 100
        print(f"f_drop_factor: {self.f_drop_factor}")
        print(f"b_drop_factor: {self.b_drop_factor}")

    def forward_backward(self, batch):
        input, target = self.parse_batch_train(batch)
    
        # features and predictions
        all_p, all_f = self.model(input, return_feature=True)

        # one-hot labels
        all_o = torch.nn.functional.one_hot(target, self.num_classes)
        
        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(self.device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.model.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.model.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, target)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(all_p, target)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        target = batch["label"]
        input = input.to(self.device)
        target = target.to(self.device)
        return input, target
