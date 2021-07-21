from torch import nn
from .attention import BertAttention
from .feed_forward import BertIntermediate, PFFOutput


class BertLayer(nn.Module):
    """Transformer block
    MultiHeadedAttention with sublayer: self_attention -> dense -> dropout -> add&&norm ->
    FFN with sublayer: dense1 -> act_fn -> dense2 -> dropout -> add&&norm
    """
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = PFFOutput(config)

    def forward(self, hidden_states, v_mask):
        attention_output = self.attention(hidden_states, v_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class BertEncoder(nn.Module):
    """Bert Encoder"""
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, v_mask, output_all_encoded_layers=False):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, v_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers
