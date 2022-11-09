import torch.nn as nn
from torch.nn import functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Conv1d(nn.Module):
    def __init__(self, n_input, n_output, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(n_input, n_output, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(n_output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class M3(nn.Module):
    """
    https://arxiv.org/pdf/1610.00087.pdf
    """

    def __init__(self, n_input=1, n_output=35, n_channel=256):
        super().__init__()

        self.conv1 = Conv1d(n_input, n_channel, 80, stride=4, padding=42)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = Conv1d(n_channel, n_channel, 3, padding=1)
        self.pool2 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(n_channel, n_output)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.squeeze()
        x = self.fc1(x)
        
        return x


class M11(nn.Module):
    """
    https://arxiv.org/pdf/1610.00087.pdf
    """
    
    def __init__(self, n_input=1, n_output=35, n_channel=64):
        super().__init__()
        
        self.conv1 = Conv1d(n_input, n_channel, 80, stride=4, padding=42)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Sequential(
            Conv1d(n_channel, n_channel, 3, padding=1),
            Conv1d(n_channel, n_channel, 3, padding=1)
        )
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Sequential(
            Conv1d(n_channel, 2 * n_channel, 3, padding=1),
            Conv1d(2 * n_channel, 2 * n_channel, 3, padding=1)
        )
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Sequential(
            Conv1d(2 * n_channel, 4 * n_channel, 3, padding=1),
            Conv1d(4 * n_channel, 4 * n_channel, 3, padding=1),
            Conv1d(4 * n_channel, 4 * n_channel, 3, padding=1)
        )
        self.pool4 = nn.MaxPool1d(4)

        self.conv5 = nn.Sequential(
            Conv1d(4 * n_channel, 8 * n_channel, 3, padding=1),
            Conv1d(8 * n_channel, 8 * n_channel, 3, padding=1)
        )

        self.fc1 = nn.Linear(8 * n_channel, n_output)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.squeeze()
        x = self.fc1(x)
        
        return x


if __name__ == "__main__":
    model = M3()
    print(f"Model size: {count_parameters(model):,}")
    # import torch
    # input = torch.rand(32, 1, 8000)
    # model(input)
