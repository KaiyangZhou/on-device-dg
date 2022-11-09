import torch
from torch.nn import functional as F


def compute_div(output_model, output_teacher, T):
    """Divergence between the output of model (student) and teacher."""
    y_model = F.log_softmax(output_model / T, dim=1)
    y_teacher = F.softmax(output_teacher / T, dim=1)
    return F.kl_div(y_model, y_teacher, reduction="batchmean") * T**2


def ood_transform(data, ood_type):
    # data: (batch, 1, sample_rate)
    if ood_type == "mixup":
        # randomly mix two audio tracks
        b = data.size(0)
        perm = torch.randperm(b)
        lmda = torch.distributions.Beta(1.0, 1.0).sample([b, 1, 1])
        lmda = lmda.to(data.device)
        return data * lmda + data[perm] * (1-lmda)
    elif ood_type == "mask":
        # randomly mask consecutive time steps
        mask = torch.ones_like(data)
        b, n = mask.size(0), mask.size(2)
        min_L, max_L = int(n * 0.1), int(n * 0.9)
        L_list = torch.randint(min_L, max_L, (b,))
        for i in range(b):
            L = L_list[i]
            t = torch.randint(0, n - L, (1,))
            mask[i, 0, t : t+L] = 0.
        return data * mask
    elif ood_type == "noise":
        # random gaussian noise
        noise = torch.randn_like(data) * 0.001
        return data + noise
    else:
        raise ValueError


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(data_loader, device, model, transform, epoch, subset="Val"):
    model.eval()
    correct = 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
    
    acc = 100. * correct / len(data_loader.dataset)
    print(f"\nTrain Epoch: {epoch}\t{subset} Accuracy: {correct}/{len(data_loader.dataset)} ({acc:.1f}%)\n")
    
    return acc


def train_erm(data_loader, device, model, transform, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):

        data = data.to(device)
        target = target.to(device)

        data = transform(data)
        output = model(data)

        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def train_kd(data_loader, device, model, teacher, transform, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):

        data = data.to(device)
        target = target.to(device)

        data = transform(data)
        output_model = model(data)
        with torch.no_grad():
            output_teacher = teacher(data)

        loss_cls = F.cross_entropy(output_model, target)
        loss_div = compute_div(output_model, output_teacher, 4.0)
        loss = 0.1 * loss_cls + 0.9 * loss_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss (cls): {loss_cls.item():.6f}\tLoss (div): {loss_div.item():.6f}")


def train_okd(data_loader, device, model, teacher, transform, optimizer, epoch, log_interval, ood_type):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):

        data = data.to(device)
        target = target.to(device)

        # ID
        data = transform(data)
        output_model = model(data)
        with torch.no_grad():
            output_teacher = teacher(data)
        loss_cls = F.cross_entropy(output_model, target)
        loss_div = compute_div(output_model, output_teacher, 4.0)

        # OOD
        data_ood = ood_transform(data, ood_type)
        output_model_ood = model(data_ood)
        with torch.no_grad():
            output_teacher_ood = teacher(data_ood)
        loss_div_ood = compute_div(output_model_ood, output_teacher_ood, 4.0)

        loss = 0.1 * loss_cls + 0.9 * (loss_div + loss_div_ood)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss (cls): {loss_cls.item():.6f}\tLoss (div): {loss_div.item():.6f}\tLoss (div_ood): {loss_div_ood.item():.6f}")


if __name__ == "__main__":
    pass
