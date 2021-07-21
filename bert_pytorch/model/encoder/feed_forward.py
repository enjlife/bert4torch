import sys
import torch.nn as nn
from ..utils import act2fn
from ..utils import BertLayerNorm


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = act2fn[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class PFFOutput(nn.Module):
    def __init__(self, config):
        super(PFFOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, x):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + x)
        return hidden_states


# class PositionwiseFeedForward(nn.Module):
#     """
#     Implements FFN and sublayer
#     dense1 -> act_fn -> dense2 -> dropout -> add&&norm"""
#
#     def __init__(self, config):
#         super(PositionwiseFeedForward, self).__init__()
#
#         self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
#         self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.act_fn = act2fn.get(config.hidden_act)
#         self.norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
#
#     def forward(self, x):
#         hidden_states = self.act_fn(self.dense1(x))
#         hidden_states = self.dropout(self.dense2(hidden_states))
#         return self.norm(self.x + hidden_states)

