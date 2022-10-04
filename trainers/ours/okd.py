import contextlib
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import mixup
from dassl.modeling.ops.utils import create_onehot

from trainers.kd.kd import KD


@contextlib.contextmanager
def freeze_model_params(model):
    try:
        for param in model.parameters():
            param.requires_grad_(False)
        yield
    finally:
        for param in model.parameters():
            param.requires_grad_(True)


def fgsm_attack(image, epsilon, adv_grad, clamps):
    # Fast Gradient Sign Method (FGSM)
    sign_adv_grad = adv_grad.sign()
    perturbed_image = image + epsilon*sign_adv_grad
    r, g, b = clamps
    perturbed_image[:, 0, :, :].clamp_(*r)
    perturbed_image[:, 1, :, :].clamp_(*g)
    perturbed_image[:, 2, :, :].clamp_(*b)
    return perturbed_image


def mixup_data(x, y):
    perm = torch.randperm(x.size(0))
    x2 = x[perm]
    y2 = y[perm]
    return mixup(x, x2, y, y2, 1.0)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(input, target, beta=1.0):
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(input.size(0))
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, target_a, target_b, lam


def get_perm(l):
    # Credit: https://github.com/ermongroup/NDA
    perm = torch.randperm(l)
    while torch.all(torch.eq(perm, torch.arange(l))):
        perm = torch.randperm(l)
    return perm


def jigsaw(data, k=2):
    # Credit: https://github.com/ermongroup/NDA
    actual_h = data.size(2)
    actual_w = data.size(3)
    h = torch.split(data, int(actual_h / k), dim=2)
    splits = []
    for i in range(k):
        splits += torch.split(h[i], int(actual_w / k), dim=3)
    fake_samples = torch.stack(splits, -1)
    for idx in range(fake_samples.size(0)):
        perm = get_perm(k * k)
        fake_samples[idx] = fake_samples[idx, :, :, :, perm]
    fake_samples = torch.split(fake_samples, 1, dim=4)
    merged = []
    for i in range(k):
        merged += [torch.cat(fake_samples[i * k:(i + 1) * k], 2)]
    fake_samples = torch.squeeze(torch.cat(merged, 3), -1)
    return fake_samples


def gaussian_noise(input, mean, std):
    noise = torch.randn(input.size(), device=input.device)
    noise = noise * std + mean
    return input + noise


@TRAINER_REGISTRY.register()
class OKD(KD):
    """Out-of-distribution Knowledge Distillation.
    
    https://arxiv.org/abs/2209.07521.
    """
    
    temperature = 4.0  # temperature
    gamma = 0.1  # weight for loss_cls
    alpha = 0.9  # weight for loss_div
    beta = 0.9  # weight for loss_div_ood

    adv_eps = 0.001  # adversarial gradient
    noise_mean = 0.0  # gaussian noise
    noise_std = 0.15  # gaussian noise
    
    def __init__(self, cfg):
        super().__init__(cfg)
        fgsm_clamps = [
            [0, 1],
            [0, 1],
            [0, 1]
        ]
        fgsm_epsilon = [self.adv_eps] * 3

        if "normalize" in cfg.INPUT.TRANSFORMS:
            fgsm_clamps = []
            pixel_mean = cfg.INPUT.PIXEL_MEAN
            pixel_std = cfg.INPUT.PIXEL_STD
            for i, (m, s) in enumerate(zip(pixel_mean, pixel_std)):
                pixel_min = - m / s
                pixel_max = (1 - m) / s
                fgsm_clamps.append([pixel_min, pixel_max])
                fgsm_epsilon[i] = fgsm_epsilon[i] * (pixel_max - pixel_min)
        
        self.fgsm_clamps = fgsm_clamps
        self.fgsm_epsilon = torch.tensor(fgsm_epsilon).view(1, 3, 1, 1).to(self.device)
        self.aug_type = cfg.OKD_AUG_TYPE

        print(f"Aug Type: {self.aug_type}")
    
    def forward_backward(self, batch):
        input, target = self.parse_batch_train(batch)

        # Create OOD data
        if self.aug_type == "adv":
            with freeze_model_params(self.model):
                input.requires_grad = True
                output_model = self.model(input)
                loss_cls = F.cross_entropy(output_model, target)
                loss_cls.backward()
                adv_grad = input.grad.data
                input_ood = fgsm_attack(input, self.fgsm_epsilon, adv_grad, self.fgsm_clamps)
        
        elif self.aug_type == "mixup":
            target_onehot = create_onehot(target, self.num_classes)
            input_ood, target_ood = mixup_data(input, target_onehot)
        
        elif self.aug_type == "cutmix":
            input_ood, target_a, target_b, lam = cutmix(input, target)
        
        elif self.aug_type == "jigsaw":
            input_ood = jigsaw(input)
        
        elif self.aug_type == "noise":
            input_ood = gaussian_noise(input, self.noise_mean, self.noise_std)
        
        elif self.aug_type == "fusion":
            # Mixup + CutMix
            target_onehot = create_onehot(target, self.num_classes)
            input_mixup, _ = mixup_data(input, target_onehot)
            input_cutmix, _, _, _ = cutmix(input, target)
            lam = torch.distributions.Beta(1.0, 1.0).sample([input.shape[0], 1, 1, 1])
            lam = lam.to(input.device)
            input_ood = input_mixup*lam + input_cutmix*(1-lam)
        
        else:
            raise NotImplementedError

        # Forward ID data
        output_model = self.model(input)
        with torch.no_grad():
            output_teacher = self.teacher(input)
        loss_cls = F.cross_entropy(output_model, target)
        loss_div = self.compute_div(output_model, output_teacher, self.temperature)

        # Forward OOD data
        output_model_ood = self.model(input_ood)
        with torch.no_grad():
            output_teacher_ood = self.teacher(input_ood)
        loss_div_ood = self.compute_div(output_model_ood, output_teacher_ood, self.temperature)

        # Loss
        loss = 0
        loss += loss_cls * self.gamma
        loss += loss_div * self.alpha
        loss += loss_div_ood * self.beta
        
        self.model_backward_and_update(loss)

        loss_summary = {}
        loss_summary["loss_cls"] = loss_cls.item()
        loss_summary["acc"] = compute_accuracy(output_model, target)[0].item()
        loss_summary["loss_div"] = loss_div.item()
        loss_summary["loss_div_ood"] = loss_div_ood.item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
