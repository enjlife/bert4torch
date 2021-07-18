import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):
    """
    p_k,2i = sin(k/10000^(2i/d))
    p_k,2i+1 = cos(k/10000^(2i/d))
    """
    def __init__(self, hidden_size):
        super(PositionalEmbedding, self).__init__()
        # self.hidden_size = hidden_size
        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))

        # register a buffer that should not to be considered a model parameter.
        # For example, BatchNorm's ``running_mean``
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        # ger: outer product of vec1 and vec2
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        # bsz not know now
        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


# class PositionalEmbedding(nn.Module):
#
#     def __init__(self, d_model, max_len=512):
#         super().__init__()
#
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False
#
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return self.pe[:, :x.size(1)]


