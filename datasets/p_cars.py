from dassl.data.datasets import DATASET_REGISTRY

from .p_air import P_Air


@DATASET_REGISTRY.register()
class P_Cars(P_Air):
    """Places-StanfordCars."""

    dataset_dir = "P-Cars"
    data_url = "https://drive.google.com/uc?id=1ZYJo_tBNh78nkAUzxZKjga31qX_nklkk"
    zip_name = "P-Cars.zip"

    def __init__(self, cfg):
        super().__init__(cfg)
