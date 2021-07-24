import torch.nn as nn
import torch
import math
from ..utils import BertLayerNorm


class BertAttention(nn.Module):
    """Completed Attention"""
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = MultiHeadedSelfAttention(config)
        self.output = AttentionOutput(config)

    def forward(self, x, attention_mask):
        self_output = self.self(x, attention_mask)
        # dense -> dropout -> add -> norm
        attention_output = self.output(self_output, x)
        return attention_output


class MultiHeadedSelfAttention(nn.Module):
    """
    MultiHeadedSelfAttention
    """
    def __init__(self, config):
        super(MultiHeadedSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of encoder "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, v_mask, unilm_mask=None,):
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw encoder scores.
        # (batch, head, pos, pos)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the encoder mask is (precomputed for all layers in BertModel forward() function)
        if unilm_mask is not None:
            attention_scores = attention_scores.masked_fill(unilm_mask == 0, -1e12)
        # Apply the encoder mask is (precomputed for all layers in BertModel forward() function)
        if v_mask is not None:
            attention_scores = attention_scores.masked_fill(v_mask == 0, -1e12)

        # Normalize the encoder scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # x: (batch, head, pos, head_hid) ->  (batch, pos, head, head_hid)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class AttentionOutput(nn.Module):
    """
    AttentionOutput -> dropout -> add -> norm
    """
    def __init__(self, config):
        super(AttentionOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, x):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + x)
        return hidden_states


# class MultiHeadedAttention(nn.Module):
#     def __init__(self, config):
#         """
#         used parameter of config:
#             num_attention_heads, attention_head_size, hidden size, num_qkv, attention_probs_dropout_prob
#
#         """
#         super(MultiHeadedAttention, self).__init__()
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError("The hidden size (%d) is not a multiple of the number of encoder "
#                              "heads (%d)" % (config.hidden_size, config.num_attention_heads))
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#
#         if hasattr(config, 'num_qkv') and (config.num_qkv > 1):
#             self.num_qkv = config.num_qkv
#         else:
#             self.num_qkv = 1
#
#         self.query = nn.Linear(config.hidden_size, self.all_head_size*self.num_qkv)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size*self.num_qkv)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size*self.num_qkv)
#         self.output_linear = nn.Linear(self.all_head_size, config.hidden_size)
#
#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#         # not know now
#         self.uni_debug_flag = True if os.getenv('UNI_DEBUG_FLAG', '') else False
#
#         if self.uni_debug_flag:
#             self.register_buffer('debug_attention_probs', torch.zeros((512, 512)))
#         if hasattr(config, 'seg_emb') and config.seg_emb:
#             self.b_q_s = nn.Parameter(torch.zeros(
#                 1, self.num_attention_heads, 1, self.attention_head_size))
#             self.seg_emb = nn.Embedding(
#                 config.type_vocab_size, self.all_head_size)
#         else:
#             self.b_q_s = None
#             self.seg_emb = None
#
#     def transpose_for_scores(self, x, mask_qkv=None):
#         """ view or reshape x and permute x to (batch, head, pos, head_hid)
#         """
#         # num_qkv not know now
#         if self.num_qkv > 1:
#             sz = x.size()[:-1] + (self.num_qkv, self.num_attention_heads, self.all_head_size)
#             # (batch, pos, num_qkv, head, head_hid)
#             x = x.view(*sz)
#             if mask_qkv is None:
#                 x = x[:, :, 0, :, :]
#             elif isinstance(mask_qkv, int):
#                 x = x[:, :, mask_qkv, :, :]
#             else:
#                 # mask_qkv: (batch, pos)
#                 if mask_qkv.size(1) > sz[1]:
#                     mask_qkv = mask_qkv[:, :sz[1]]
#                 # x: (batch, pos, num_qkv, head, head_hid) -> x: (batch, pos, head, head_hid)
#                 x = x.gather(2, mask_qkv.view(sz[0], sz[1], 1, 1, 1).expand(
#                     sz[0], sz[1], 1, sz[3], sz[4])).squeeze(2)
#         else:
#             sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#             # (batch, pos, head, head_hid)
#             x = x.view(*sz)
#         # (batch, head, pos, head_hid)
#         return x.permute(0, 2, 1, 3)
#
#     def forward(self, x, v_mask, unilm_mask=None, history_states=None, mask_qkv=None, seg_ids=None):
#         """
#         x: input hidden states
#         v_mask: encoder mask for padding
#         unilm_mask: special mask for some LM models, i.e. unilm
#         """
#         # history_states for decoder
#         if history_states is None:
#             mixed_query_layer = self.query(x)
#             mixed_key_layer = self.key(x)
#             mixed_value_layer = self.value(x)
#         else:
#             x_states = torch.cat((history_states, x), dim=1)
#             mixed_query_layer = self.query(x)
#             mixed_key_layer = self.key(x_states)
#             mixed_value_layer = self.value(x_states)
#
#         query_layer = self.transpose_for_scores(mixed_query_layer, mask_qkv)
#         key_layer = self.transpose_for_scores(mixed_key_layer, mask_qkv)
#         value_layer = self.transpose_for_scores(mixed_value_layer, mask_qkv)
#
#         # Take the dot product between "query" and "key" to get the raw encoder scores.
#         # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         # (batch, head, pos, pos)
#         attention_scores = torch.matmul(
#             query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
#         # not know now
#         if self.seg_emb is not None:
#             seg_rep = self.seg_emb(seg_ids)
#             # (batch, pos, head, head_hid)
#             seg_rep = seg_rep.view(seg_rep.size(0), seg_rep.size(1),
#                                    self.num_attention_heads, self.attention_head_size)
#             qs = torch.einsum('bnih,bjnh->bnij', query_layer+self.b_q_s, seg_rep)
#             attention_scores = attention_scores + qs
#
#         if unilm_mask is not None:
#             attention_scores = attention_scores.masked_fill(unilm_mask == 0, -1e12)
#         # Apply the encoder mask is (precomputed for all layers in BertModel forward() function)
#         if v_mask is not None:
#             attention_scores = attention_scores.masked_fill(v_mask == 0, -1e12)
#
#         # Normalize the encoder scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)
#
#         if self.uni_debug_flag:
#             _pos = attention_probs.size(-1)
#             self.debug_attention_probs[:_pos, :_pos].copy_(attention_probs[0].mean(0).view(_pos, _pos))
#
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)
#
#         context_layer = torch.matmul(attention_probs, value_layer)
#         # x: (batch, head, pos, head_hid) ->  (batch, pos, head, head_hid)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         # x: (batch, pos, head, head_hid) -> (batch, pos, all_head_size)
#         res_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*res_shape)
#
#         return self.output_linear(context_layer)


# class Attention(nn.Module):
#     """
#     Compute 'Scaled Dot Product Attention
#     """
#
#     def forward(self, query, key, value, mask=None, dropout=None):
#         scores = torch.matmul(query, key.transpose(-2, -1)) \
#                  / math.sqrt(query.size(-1))
#
#         if mask is not None:
#             # masked [batch_size, h, seq_len, d_k]
#             scores = scores.masked_fill(mask == 0, -1e9)
#
#         p_attn = F.softmax(scores, dim=-1)
#
#         if dropout is not None:
#             p_attn = dropout(p_attn)
#
#         return torch.matmul(p_attn, value), p_attn
#
#
# class MultiHeadedAttention(nn.Module):
#     """
#     Take in model size and number of heads.
#     """
#
#     def __init__(self, h, d_model, dropout=0.1):
#         super().__init__()
#         assert d_model % h == 0
#
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#
#         self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
#         self.output_linear = nn.Linear(d_model, d_model)
#         self.encoder = Attention()
#
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)
#
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
#                              for l, x in zip(self.linear_layers, (query, key, value))]
#
#         # 2) Apply encoder on all the projected vectors in batch.
#         x, attn = self.encoder(query, key, value, mask=mask, dropout=self.dropout)
#
#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
#
#         return self.output_linear(x)

