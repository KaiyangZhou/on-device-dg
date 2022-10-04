from dassl.data.datasets import DATASET_REGISTRY

from .p_air import P_Air


@DATASET_REGISTRY.register()
class P_UCF(P_Air):
    """Places-UCF101."""

    dataset_dir = "P-UCF"
    data_url = "https://drive.google.com/uc?id=1_lu3k9v18VFiM5JxNFMvvoUMcZDihPh8"
    zip_name = "P-UCF.zip"

    def __init__(self, cfg):
        super().__init__(cfg)
