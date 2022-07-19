import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F

class TIMutaNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_hid_layers, dropout, device):
        '''
        in_dim: Input dimension
        hid_dim: Number of neuron in hidden layers
        out_dim: Output dim (1 for this paper due to binary cls between IPMN/PAD)
        n_hid_layers: Number of neural layers (Adaptive)
        dropout: Dropout rate for regularization
        '''
        super(TIMutaNet,self).__init__()
        self.in_layer = nn.Linear(in_features = in_dim, out_features = hid_dim)
        self.hid_layers = [nn.Sequential(nn.LayerNorm(hid_dim),nn.Linear(hid_dim,hid_dim),
                            nn.Dropout(dropout), nn.ReLU()).to(device)
                            for _ in range(n_hid_layers)]
        self.out_layer = nn.Linear(in_features = hid_dim, out_features = out_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.in_layer(x)
        for f in self.hid_layers:
            x = f(x) + x
        x = self.out_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x
