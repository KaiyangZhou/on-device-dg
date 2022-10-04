import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, OfficeHomeDG


@DATASET_REGISTRY.register()
class OfficeHomeDG2(DatasetBase):
    """OfficeHomeDG."""

    dataset_dir = "office_home_dg"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        dataset = OfficeHomeDG(cfg)
        train, val, test = dataset.train_x, dataset.val, dataset.test
        
        env = cfg.DATASET.ENV
        if env != "full":
            raise NotImplementedError

        super().__init__(train_x=train, val=val, test=test)