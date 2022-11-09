"""
Credit: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
"""
import os
import sys
import argparse

import torch
import torch.optim as optim
import torchaudio

from dassl.utils import setup_logger, set_random_seed

from model import M3, M11, count_parameters
from dataset import build_loaders
from core import test, train_erm, train_kd, train_okd


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--split_id", type=int, default=1, choices=[1, 2, 3])
parser.add_argument("-m", "--model_name", type=str, default="M3")
parser.add_argument("-b", "--batch_size", type=int, default=256)
parser.add_argument("-n", "--n_epoch", type=int, default=60)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--alg", type=str, default="erm", choices=["erm", "kd", "okd"])
parser.add_argument("--ood_type", type=str, default="mixup", choices=["mixup", "mask", "noise"])
parser.add_argument("--log_interval", type=int, default=20)
parser.add_argument("-o", "--output_dir", type=str, default="output")
args = parser.parse_args()
print(args)

split_id = args.split_id
model_name = args.model_name
batch_size = args.batch_size
n_epoch = args.n_epoch
lr = args.lr
wd = args.wd
alg = args.alg
ood_type = args.ood_type
log_interval = args.log_interval
output_dir = args.output_dir

if os.path.exists(output_dir):
    print(f"Results already exist at {output_dir}")
    sys.exit()

set_random_seed(split_id)
setup_logger(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_name == "M3":
    model = M3()
elif model_name == "M11":
    model = M11()
else:
    raise ValueError
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch)
print(f"Model size: {count_parameters(model):,}")

teacher = None
if alg in ["kd", "okd"]:
    teacher = M11()
    teacher.to(device)
    teacher.load_state_dict(torch.load(f"output/erm/M11/seed{split_id}/model.pt"))
    teacher.eval()
    print(f"Teacher size: {count_parameters(teacher):,}")

train_loader, val_loader, test_loader = build_loaders(split_id, batch_size)
transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
transform = transform.to(device)

best_acc = 0
best_epoch = 1
best_model = None
acc_list = []

print("\n*** Start training ***\n")

for epoch in range(1, n_epoch + 1):
    if alg == "erm":
        train_erm(train_loader, device, model, transform, optimizer, epoch, log_interval)
    elif alg == "kd":
        train_kd(train_loader, device, model, teacher, transform, optimizer, epoch, log_interval)
    elif alg == "okd":
        train_okd(train_loader, device, model, teacher, transform, optimizer, epoch, log_interval, ood_type)
    else:
        raise ValueError
    
    acc = test(val_loader, device, model, transform, epoch, "Val")
    acc_list.append((epoch, acc))
    
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        best_model = model.state_dict()
    
    scheduler.step()

print("\nDone training")
for epoch, acc in acc_list:
    print(f"Epoch: {epoch}\tVal Accuracy: {acc:.1f}%")

if best_model is not None:
    print(f"Loading weights at epoch {best_epoch}")
    model.load_state_dict(best_model)

test(test_loader, device, model, transform, epoch, "Test")
torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
