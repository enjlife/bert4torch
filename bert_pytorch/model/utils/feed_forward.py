import torch.nn as nn
from .activations import act2fn
from .layer_norm import LayerNorm


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN and sublayer
    dense1 -> act_fn -> dense2 -> dropout -> add&&norm"""

    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = act2fn.get(config.hidden_act)
        self.norm = LayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, x):
        hidden_states = self.act_fn(self.dense1(x))
        hidden_states = self.dropout(self.dense2(hidden_states))
        return self.norm(self.x + hidden_states)
