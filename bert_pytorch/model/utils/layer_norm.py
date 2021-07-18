import torch.nn as nn
import torch


# TODO: conditional layer normalization
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()

        # in tf: weight: gamma, bias: beta
        # so when load weight in tf, need exchange
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.variance_epsilon)

        return self.weight * x + self.bias
