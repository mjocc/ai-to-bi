from torch_geometric.nn import TransformerConv
import torch.nn as nn
import torch

class GNNModel(nn.Module):
    def __init__(self, channels=(3, 16, 32, 32)):
        super(GNNModel, self).__init__()
        down_blocks = []
        for i in range(len(channels)-1):
            down_blocks.append(nn.Conv2d(channels[i], channels[i+1], 3, 1, 1))
            down_blocks.append(nn.BatchNorm2d(channels[i+1]))
            down_blocks.append(nn.MaxPool2d(2))
            down_blocks.append(nn.Dropout(p=0.2))
            down_blocks.append(nn.ReLU())
        down_blocks.append(nn.Flatten())
        self.down = nn.Sequential(*down_blocks)
        self.expected_dim = 6*6*channels[-1]
        self.coords_in = nn.Linear(2, self.expected_dim)
        self.out = nn.Sequential(nn.Linear(self.expected_dim, 20), nn.Linear(20, 2))
    
    def forward(self, images):
        processed = self.down(images)
        assert processed.ndim == 2 and processed.shape[-1] == self.expected_dim
        processed = processed
        return self.out(processed)