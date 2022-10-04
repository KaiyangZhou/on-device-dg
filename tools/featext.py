"""
Extract features using PlacesCNN.
"""
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision.models as models
from torchvision import transforms as T
from torch.nn import functional as F

from dassl.config import get_cfg_default
from dassl.data.datasets import build_dataset


model_files = {
    "vit_large_patch16": "mae_finetune_places365_vit_large_epoch26.pth"
}
arch = "vit_large_patch16"
model_file = model_files[arch]
assert os.path.exists(model_file)
checkpoint_model = torch.load(model_file)
print(f"Loaded model weights from {model_file}")
model = models_vit.__dict__[arch]()
model.load_state_dict(checkpoint_model, strict=False)
model.eval()
model = model.cuda()

# load the image transformer
t = []
# maintain same ratio w.r.t. 224 images
# follow https://github.com/facebookresearch/mae/blob/main/util/datasets.py
t.append(T.Resize(224, interpolation=Image.BICUBIC))
t.append(T.CenterCrop(224))
t.append(T.ToTensor())
t.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
center_crop = T.Compose(t)

# Input dataset name
dataset_name = sys.argv[1]

save_dir = "../data/datasets/" + dataset_name + "/features_placesvit"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    sys.exit()

"""
The structure should be:
dataset/
    class-1/
        image-001.jpg
        image-002.jpg
    class-2/
        ...
    ...
"""
image_dir = "../data/" + dataset_name + "/images"
class_dirs = os.listdir(image_dir)
for class_dir in class_dirs:
    print(f"Processing {class_dir}")
    examples = os.listdir(os.path.join(image_dir, class_dir))
    examples = [os.path.join(class_dir, example) for example in examples]
    
    imgs = []
    for example in examples:
        path = os.path.join(image_dir, example)
        img = Image.open(path).convert("RGB")
        img = center_crop(img)
        imgs.append(img)
    
    imgs = torch.stack(imgs).cuda()
    with torch.no_grad():
        features = model(imgs)
    features = features.cpu().numpy().astype(np.float32)
    
    save_file = os.path.join(save_dir, class_dir)
    np.savez(save_file, examples=examples, features=features)
