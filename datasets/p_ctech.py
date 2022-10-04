from dassl.data.datasets import DATASET_REGISTRY

from .p_air import P_Air


@DATASET_REGISTRY.register()
class P_Ctech(P_Air):
    """Places-Caltech."""

    dataset_dir = "P-Ctech"
    data_url = "https://drive.google.com/uc?id=1fmwrXHglxAU8EbBPDxHguA1zypgqW5Mi"
    zip_name = "P-Ctech.zip"

    def __init__(self, cfg):
        super().__init__(cfg)
