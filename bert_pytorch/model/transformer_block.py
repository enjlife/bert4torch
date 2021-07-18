import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import LayerNorm, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    MultiHeadedAttention with sublayer: self_attention -> dense -> dropout -> add&&norm ->
    FFN with sublayer: dense1 -> act_fn -> dense2 -> dropout -> add&&norm
    """

    def __init__(self, config):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadedAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = LayerNorm(config.hidden_size)
        self.feed_forward = PositionwiseFeedForward(config)

    def forward(self, x, mask):
        hidden_states = self.attention(x, v_mask=mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states + x)

        return self.feed_forward(hidden_states)
