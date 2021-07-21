import torch
import torch.nn as nn
from packaging import version
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
from ..utils import BertLayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# class BERTEmbedding(nn.Module):
#     """
#     BERT Embedding which is consisted with under features
#         1. TokenEmbedding : normal embedding matrix
#         2. PositionalEmbedding : adding positional information using sin, cos
#         2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
#
#         sum of all these features are output of BERTEmbedding
#     """
#
#     def __init__(self, vocab_size, embed_size, dropout=0.1):
#         """
#         :param vocab_size: total vocab size
#         :param embed_size: embedding size of token embedding
#         :param dropout: dropout rate
#         """
#         super().__init__()
#         self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
#         self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
#         self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
#         self.dropout = nn.Dropout(p=dropout)
#         self.embed_size = embed_size
#
#     def forward(self, sequence, segment_label):
#         x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
#         return self.dropout(x)