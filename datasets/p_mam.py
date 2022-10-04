from dassl.data.datasets import DATASET_REGISTRY

from .p_air import P_Air


@DATASET_REGISTRY.register()
class P_Mam(P_Air):
    """Places-Mamal."""

    dataset_dir = "P-Mam"
    data_url = "https://drive.google.com/uc?id=1xCjidzliVk8W5GDWJhZlJVomtWU6EOr0"
    zip_name = "P-Mam.zip"

    def __init__(self, cfg):
        super().__init__(cfg)
