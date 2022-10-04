from dassl.data.datasets import DATASET_REGISTRY

from .p_air import P_Air


@DATASET_REGISTRY.register()
class P_Pets(P_Air):
    """Places-OxfordPets."""

    dataset_dir = "P-Pets"
    data_url = "https://drive.google.com/uc?id=1IgMKByvuMS-ZkgxhlpHYv06HNLAcWslz"
    zip_name = "P-Pets.zip"

    def __init__(self, cfg):
        super().__init__(cfg)
