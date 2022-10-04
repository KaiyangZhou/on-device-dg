import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, PACS


@DATASET_REGISTRY.register()
class PACS2(DatasetBase):
    """PACS."""

    dataset_dir = "pacs"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        dataset = PACS(cfg)
        train, val, test = dataset.train_x, dataset.val, dataset.test
        
        env = cfg.DATASET.ENV
        if env != "full":
            raise NotImplementedError

        super().__init__(train_x=train, val=val, test=test)