import json
import os.path as osp

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class P_Air(DatasetBase):
    """Places-FGVCAircraft."""

    dataset_dir = "P-Air"
    data_url = "https://drive.google.com/uc?id=1qHn1FccGwkJQOB_VN2v2OKwN6nhQmcDg"
    zip_name = "P-Air.zip"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "zhou_splits")

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, self.zip_name)
            self.download_data(self.data_url, dst, from_gdrive=True)

        seed = cfg.SEED  # indicates the version of split
        split_name = f"random_v{cfg.SEED}_{cfg.DATASET.ENV}.json"
        split_path = osp.join(self.split_dir, split_name)
        assert osp.exists(split_path), split_path
        print(f"Reading splits from {split_path}")
        train, val, test = self.read_split(split_path, self.image_dir)

        super().__init__(train_x=train, val=val, test=test)
    
    @staticmethod
    def read_split(split_path, image_dir):

        def _construct(imnames, label, classname):
            return [
                Datum(
                    impath=osp.join(image_dir, imname),
                    label=label,
                    classname=classname)
                for imname in imnames
            ]
        
        with open(split_path, "r") as file:
            split = json.load(file)
        
        train, val, test = [], [], []
        classnames = list(split.keys())
        classnames.sort()
        
        for label, classname in enumerate(classnames):
            
            imnames_train = split[classname]["train"]
            imnames_val = split[classname]["val"]
            imnames_test = split[classname]["test"]

            train.extend(_construct(imnames_train, label, classname))
            val.extend(_construct(imnames_val, label, classname))
            test.extend(_construct(imnames_test, label, classname))
        
        return train, val, test