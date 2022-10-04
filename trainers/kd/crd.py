"""
Credit: https://github.com/HobbitLong/RepDistiller
"""
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset

from dassl.engine import TRAINER_REGISTRY
from dassl.data import DataManager
from dassl.data.datasets import DatasetBase
from dassl.data.transforms import INTERPOLATION_MODES
from dassl.utils import count_num_param, read_image
from dassl.optim import build_optimizer, build_lr_scheduler

from .kd import KD
from .crd_tools import CRDLoss

# CRD-related hyper-parameters
MODE = "exact"  # {"exact", "relax"}
NCE_K = 16384


class DatasetWrapper2(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False, percent=1.0):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

        ######
        # CRD
        ######
        num_classes = DatasetBase.get_num_classes(data_source)
        num_samples = len(data_source)
        
        self.cls_positive = [[] for i in range(num_classes)]
        for i, item in enumerate(data_source):
            self.cls_positive[item.label].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [
                np.random.permutation(self.cls_negative[i])[0:n]
                for i in range(num_classes)
            ]

        self.cls_positive = np.asarray(self.cls_positive, dtype=object)
        self.cls_negative = np.asarray(self.cls_negative, dtype=object)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        target = item.label

        output = {
            "label": target,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation
        
        if self.is_train:
            # Sample contrastive examples
            if MODE == "exact":
                pos_idx = idx
            elif MODE == "relax":
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(MODE)
            replace = True if NCE_K > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], NCE_K, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            output["contrast_idx"] = sample_idx

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


@TRAINER_REGISTRY.register()
class CRD(KD):
    """
    Contrastive Representation Distillation.
    """

    beta = 0.1

    def __init__(self, cfg):
        super().__init__(cfg)
        self.build_crd()
    
    def build_crd(self):
        cfg = self.cfg
        
        input = torch.rand(2, 3, *cfg.INPUT.SIZE)
        input = input.to(self.device)
        with torch.no_grad():
            _, feature_model = self.model(input, output_f_mid=True)
            _, feature_teacher = self.teacher(input, output_f_mid=True)

        """
        opt["s_dim"]: the dimension of student's feature
        opt["t_dim"]: the dimension of teacher's feature
        opt["feat_dim"]: the dimension of the projection space
        opt["nce_k"]: number of negatives paired with each positive
        opt["nce_t"]: temperature parameter for softmax
        opt["nce_m"]: the momentum for updating the memory buffer
        opt["n_data"]: the number of samples in the training set, therefor the memory buffer is: opt["n_data"] x opt["feat_dim"]
        """
        opt = {}
        opt["s_dim"] = feature_model[-1].shape[1]
        opt["t_dim"] = feature_teacher[-1].shape[1]
        opt["feat_dim"] = 128
        opt["nce_k"] = NCE_K
        opt["nce_t"] = 0.1
        opt["nce_m"] = 0.9
        opt["n_data"] = len(self.dm.dataset.train_x)
        
        print("Building crd-learner")
        self.crd_learner = CRDLoss(opt)
        self.crd_learner.to(self.device)
        print(f"# params: {count_num_param(self.crd_learner):,}")
        self.optim_crd = build_optimizer(self.crd_learner, cfg.OPTIM)
        self.sched_crd = build_lr_scheduler(self.optim_crd, cfg.OPTIM)
        self.register_model("crd_learner", self.crd_learner, self.optim_crd, self.sched_crd)
    
    def build_data_loader(self):
        dm = DataManager(self.cfg, dataset_wrapper=DatasetWrapper2)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
    
    def compute_kd(self, feature_model, feature_teacher, batch):
        idx = batch["index"].to(self.device)
        contrast_idx = batch["contrast_idx"].to(self.device)
        f_m = feature_model[-1]
        f_t = feature_teacher[-1]
        return self.crd_learner(f_m, f_t, idx, contrast_idx)
