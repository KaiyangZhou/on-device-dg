# On-Device Domain Generalization

## Overview

This repo contains the source code of our project, "[On-Device Domain Generalization](https://arxiv.org/abs/2209.07521)," which studies how to improve tiny neural networks' domain generalization (DG) performance, specifically for mobile DG applications. In the paper, we present a systematic study from which we find that [knolwedge distillation](https://arxiv.org/abs/1503.02531) outperforms commonly-used DG methods by a large margin under the on-device DG setting. We further propose a simple idea, called **out-of-distribution knolwedge distillation (OKD)**, which extends KD by teaching the student how the teacher handles out-of-distribution data synthesized via data augmentations. We also provide a new suite of DG datasets, named **DOSCO-2k**, which are built on top of existing vision datasets (much more diverse than existing DG datasets) by synthesizing contextual domain shift using a neural network pretrained on the [Places](http://places2.csail.mit.edu/) dataset.

## Updates

- **Oct 2022**: Release of source code.

## Get Started

### 1. Environment Setup

This code is built on top of the awesome toolbox, [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `on-device-dg/` to install a few more packages (remember to activate the `dassl` environment via `conda activate dassl` before installing the new packages).

### 2. Datasets and Models

We suggest you download and put all datasets under the same folder, e.g., `on-device-dg/data/`.

- **PACS & OfficeHome**: These two datasets are small (both around 200MB) so we suggest you directly run the code, which will automatically download and preprocess the datasets.
- **DOSCO-2k**: All datasets from the DOSCO benchmark can be downloaded automatically once you run the code (like PACS and OfficeHome). But we suggest you manually download them first. They can be downloaded from this google drive [link](https://drive.google.com/drive/folders/1QJTz4vJ4Jta6Co6UHYmVnXJUGK1H9-G3?usp=sharing).
- **Pretrained teacher models (ResNet50)**: The pretrained ERM models based on ResNet50, i.e., *KD's teacher* as reported in the paper, can be downloaded [here](https://drive.google.com/file/d/1x7jk8ibhlEsh4RQwrepK-mbg5NflLnQG/view?usp=sharing). Please download and extract the file under `on-device-dg/`. To reproduce the results of KD and OKD, you should use these pretrained teacher models.
- **PlacesViT**: The model weights can be downloaded [here](https://drive.google.com/file/d/1__940fYMzzObU48JP3cveVHbUeexhk23/view?usp=sharing). Please put the weights under `on-device-dg/tools/`. The feature extraction code is provided in `on-device-dg/tools/featext.py`.

### 3. How to Run

The running scripts are provided in `on-device-dg/scripts/`:
- `generic.sh`: This can fit most trainers like `Vanilla`.
- `kd.sh`: This is used for those KD-based trainers in `on-device-dg/trainers/` (except OKD).
- `okd.sh` This is used for OKD, which mainly differs from `kd.sh` in the `Aug` argument (it chooses which augmentation method to use for the OOD data generator).

The `DATA_ROOT` argument is set to `./data/` by default. Feel free to change the path.

Below are the example commands used to reproduce the results on DOSCO-2k's P-Air using MobileNetV3-Small (should be run under `on-device-dg/`):
- **ERM**: `bash scripts/generic.sh Vanilla p_air mobilenet_v3_small 2k`
- **RSC**: `bash scripts/generic.sh RSC p_air mobilenet_v3_small 2k`
- **MixStyle**: `bash scripts/generic.sh Vanilla p_air mobilenet_v3_small_ms_l12.yaml 2k`
- **EFDMix**: `bash scripts/generic.sh Vanilla p_air mobilenet_v3_small_efdmix_l12.yaml 2k`
- **KD**: `bash scripts/kd.sh KD p_air mobilenet_v3_small 2k`
- **OKD**: `bash scripts/okd.sh OKD fusion p_air mobilenet_v3_small 2k`

Some notes:
- MixStyle and EFDMix use the same trainer as ERM, i.e., `Vanilla`.
- To use a different dataset, simply change `p_air`. Note that the dataset names should match the file names in `on-device-dg/configs/datasets/`, such as `p_cars` for `P-Cars` and `p_ctech` for `P-Ctech`.
- To use a different architecture like MobileNetV2-Tiny or MCUNet studied in the paper, simply change `mobilenet_v3_small` to `mobilenet_v2_tiny` or `mcunet`. (The model names should match the file names in `on-device-dg/configs/hparam`.)
- To reproduce the results on PACS and OfficeHome, you need to (i) change `p_air` to `pacs` or `oh`, (ii) change `2k` to `full`, and (iii) add an index number from `{1, 2, 3, 4}` at the end of the argument list. Say you want to run OKD on PACS, which has four settings (each using one of the four domains as the test domain), the command template is `bash scripts/okd.sh OKD fusion pacs mobilenet_v3_small 2k {TIDX}` where `TIDX = 1/2/3/4`.
- After you obtain the results of three seeds, you can use `parse_test_res.py` to automatically compute the average results. You can give a quick try: say you have downloaded the pretrained teacher models at `on-device-dg/pretrained`, run `python parse_test_res.py pretrained/Vanilla/p_air/env_2k/resnet50/` to get the average results for the P-Air dataset (basically `../resnet50/` should contain three seed folders each containing a `log.txt` file). Note that for PACS and OfficeHome, the `../resnet50/` folder contains four sets of results each corresponding to a test domain, you need to use `python parse_test_res.py pretrained/Vanilla/pacs/env_full/resnet50/ --multi-exp`.

## Citation

```bash
@article{zhou2022device,
  title={On-Device Domain Generalization},
  author={Zhou, Kaiyang and Zhang, Yuanhan and Zang, Yuhang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  journal={arXiv preprint arXiv:2209.07521},
  year={2022}
}
```
