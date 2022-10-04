"""
Credit: https://github.com/HobbitLong/RepDistiller
"""
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.modeling import build_backbone, build_head
from dassl.optim import build_optimizer, build_lr_scheduler


class SimpleNet2(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, output_f_mid=False):
        if output_f_mid:
            f, f_mid = self.backbone(x, output_f_mid=output_f_mid)
            f_mid.append(f)  # include the top-layer features
        else:
            f = self.backbone(x)
        
        if self.head is not None:
            f = self.head(f)
        
        if self.classifier is None:
            y = f
        else:
            y = self.classifier(f)

        if output_f_mid:
            return y, f_mid
        else:
            return y


@TRAINER_REGISTRY.register()
class KD(TrainerX):
    """
    Knowledge Distillation.
    """

    hparam_table = []
    temperature = 4.0  # temperature
    gamma = 0.1  # weight for loss_cls
    alpha = 0.9  # weight for loss_div
    beta = 0.0  # weight for loss_kd

    def __init__(self, cfg):
        super().__init__(cfg)

        self.hparam_table.append(["temperature", self.temperature])
        self.hparam_table.append(["gamma", self.gamma])
        self.hparam_table.append(["alpha", self.alpha])
        self.hparam_table.append(["beta", self.beta])
        print(tabulate(self.hparam_table))
    
    def build_model(self):
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet2(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        print("Building teacher")
        self.teacher = SimpleNet2(cfg, cfg.TEACHER, self.num_classes)
        load_pretrained_weights(self.teacher, cfg.TEACHER.INIT_WEIGHTS)
        self.teacher.to(self.device)
        self.teacher.eval()
        print(f"# params: {count_num_param(self.teacher):,}")

    def forward_backward(self, batch):
        input, target = self.parse_batch_train(batch)

        output_model, feature_model = self.model(input, output_f_mid=True)
        with torch.no_grad():
            output_teacher, feature_teacher = self.teacher(input, output_f_mid=True)
        
        loss_cls = F.cross_entropy(output_model, target)
        loss_div = self.compute_div(output_model, output_teacher, self.temperature)
        loss_kd = self.compute_kd(feature_model, feature_teacher, batch)

        loss = 0
        loss += loss_cls * self.gamma
        loss += loss_div * self.alpha
        if loss_kd is not None:
            loss += loss_kd * self.beta
        self.model_backward_and_update(loss)

        loss_summary = {}
        loss_summary["loss_cls"] = loss_cls.item()
        loss_summary["acc"] = compute_accuracy(output_model, target)[0].item()
        loss_summary["loss_div"] = loss_div.item()
        if loss_kd is not None:
            loss_summary["loss_kd"] = loss_kd.item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        target = batch["label"]
        input = input.to(self.device)
        target = target.to(self.device)
        return input, target
    
    @staticmethod
    def compute_div(output_model, output_teacher, T):
        """Divergence between the output of model (student) and teacher."""
        y_model = F.log_softmax(output_model / T, dim=1)
        y_teacher = F.softmax(output_teacher / T, dim=1)
        return F.kl_div(y_model, y_teacher, reduction="batchmean") * T**2
    
    def compute_kd(self, feature_model, feature_teacher, batch):
        """Specialized KD loss."""
        return None
