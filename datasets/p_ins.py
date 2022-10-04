from dassl.data.datasets import DATASET_REGISTRY

from .p_air import P_Air


@DATASET_REGISTRY.register()
class P_Ins(P_Air):
    """Places-Instrumentality."""

    dataset_dir = "P-Ins"
    data_url = "https://drive.google.com/uc?id=1A99b5xhzWUYbFIqOItFbB-MRmrPm4Yql"
    zip_name = "P-Ins.zip"

    def __init__(self, cfg):
        super().__init__(cfg)
