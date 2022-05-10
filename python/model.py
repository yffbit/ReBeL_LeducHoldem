from turtle import forward
import torch
from torch import nn

def build_mlp(input_dim, hidden_dims, output_dim=None, activation=None, dropout=0, use_layer_norm=False):
    net = []
    if activation is None:
        activation = nn.GELU()
    last_dim = input_dim
    for dim in hidden_dims:
        net.append(nn.Linear(last_dim, dim))
        if use_layer_norm:
            net.append(nn.LayerNorm(dim))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
        last_dim = dim
    if output_dim is not None:
        net.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*net)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.body = build_mlp(input_dim, hidden_dims, use_layer_norm=True)
        last_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        self.out = nn.Linear(last_dim, output_dim)
        # 使初始输出接近0
        out_coef = 0.01
        with torch.no_grad():
            self.out.weight.data *= out_coef
            self.out.bias *= out_coef
    
    def forward(self, x):
        return self.out(self.body(x))