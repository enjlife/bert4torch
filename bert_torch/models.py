# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import copy
import json
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .layers import BertEmbeddings, BertLayerNorm, BertEncoder, Pooler, BertNSPHead, BertPreTrainingHeads, BertMLMHead

from .utils import act2fn, get_logger, load_tf_weights_in_bert


logger = get_logger(__name__)


class BertConfig(object):
    """bert的config配置类
    """
    def __init__(self,
                 vocab_size_or_config_json_file,  # config文件
                 pad_token_id=0,
                 hidden_size=768,  # 隐藏层大小 encoder,pooler
                 num_hidden_layers=12,  # 隐藏层layer的层数
                 num_attention_heads=12,  # attention头数
                 intermediate_size=3072,  # intermediate的size
                 hidden_act="gelu",  # encoder,pooler的激活函数
                 hidden_dropout_prob=0.1,  # embeddings,encoder,pooler的全连接层drop比例
                 attention_probs_dropout_prob=0.1,  # encoder的drop比例
                 layer_norm_eps=1e-12,
                 max_position_embeddings=512,  # 输入的最大长度
                 type_vocab_size=2,  # 输入的文本段落数
                 relax_projection=0,
                 new_pos_ids=False,
                 initializer_range=0.02,  # 截断正态分布的标准差
                 task_idx=None,
                 fp32_embedding=False,
                 ffn_type=0,
                 label_smoothing=None,
                 num_qkv=0,
                 seg_emb=False,
                 with_unilm=False,
                 last_fn='tanh',
                 ):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
            # 除了config文件, 补充的参数如下
            self.with_unilm = with_unilm
            self.last_fn = last_fn

        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.pad_token_id = pad_token_id
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.layer_norm_eps = layer_norm_eps
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.new_pos_ids = new_pos_ids
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.ffn_type = ffn_type
            self.label_smoothing = label_smoothing
            self.num_qkv = num_qkv
            self.seg_emb = seg_emb
            # 补充的参数
            self.with_unilm = with_unilm
            self.last_fn = last_fn
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertPreTrainedModel(nn.Module):
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    BERT_CONFIG_NAME = 'bert_config.json'
    TF_WEIGHTS_NAME = 'bert_model.ckpt'
    """
    BERT模型父类
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *inputs, **kwargs):
        state_dict = kwargs.get('state_dict', None)  # 传入模型的参数
        kwargs.pop('state_dict', None)
        from_tf = kwargs.get('from_tf', False)  # tf checkpoint
        kwargs.pop('from_tf', None)
        # Load config
        config_file = os.path.join(pretrained_model_path, cls.CONFIG_NAME)
        if not os.path.exists(config_file):
            config_file = os.path.join(pretrained_model_path, cls.BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        # logger.info("Model config \n{}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(pretrained_model_path, cls.WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(pretrained_model_path, cls.TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys, new_keys = [], []
        for key in state_dict.keys():
            new_key = None
            # Layer Norm
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            # add for uer bert pretrained_model
            if key.startswith('embedding'):
                new_key = 'bert.' + key.replace('embedding', 'embeddings')
            elif key.startswith('encoder'):
                new_key = 'bert.' + key.replace()
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.warning("Weights of {} not initialized from pretrained model: {}".format(
                            model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.warning("Weights from pretrained model not used in {}: {}".format(
                            model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                model.__class__.__name__, "\n\t".join(error_msgs)))
        return model

    def freeze_module(self, module_name):
        module = self.get_submodule(module_name)
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module_name):
        module = self.get_submodule(module_name)
        for param in module.parameters():
            param.requires_grad = True

    def get_submodule(self, target):
        if target == "":
            return self
        atoms = target.split(".")
        mod: torch.nn.Module = self
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                     "attribute `" + item + "`")
            mod = getattr(mod, item)
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not "
                                     "an nn.Module")
        return mod


class BertModel(BertPreTrainedModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.embeddings = BertEmbeddings(config)  # sum of positional, segment, token embeddings
        self.encoder = BertEncoder(config)
        self.pooler = Pooler(config)  # if config.with_pool or config.with_nsp else None
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask=None, output_all_encoded_layers=False, first_last_avg=False):
        if attention_mask is None:
            # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
            # mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)
            attention_mask = (input_ids > 0).unsqueeze(1).unsqueeze(1)  # mask -> [batch_size, 1, 1, seq_len]
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if self.config.with_unilm:
            # TODO add additional mask on attention mask
            pass
        emb_output = self.embeddings(input_ids, token_type_ids)
        encoder_layers = self.encoder(emb_output, attention_mask, output_all_encoded_layers=output_all_encoded_layers)
        # 输出第一层和最后一层的向量
        if first_last_avg:
            encoder_layers = (encoder_layers[0] + encoder_layers[-1]) / 2
        elif not output_all_encoded_layers:
            encoder_layers = encoder_layers[-1]
        pooled_output = self.pooler(encoder_layers)

        return encoder_layers, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    """MLM + NSP
    输入:
    `masked_lm_labels`: [batch_size, sequence_length] 包含索引[-1, 0, ..., vocab_size] label=-1不计算损失
    `next_sentence_label`: [batch_size] 0-连续的句子 1-随机的句子
    注意：第二句中seg_ids填充为0
    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            # CrossEntropyLoss = log_softmax + nLLLoss
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # 索引为-1不计算损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """MLM: 参考BertForPreTraining MLM部分
    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(BertPreTrainedModel):
    """NSP 参考 BertForPreTraining的NSP部分
    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForNSPSequenceClassification(BertPreTrainedModel):
    """NSP 参考 BertForPreTraining的NSP部分
    """
    def __init__(self, config, num_labels2):
        super(BertForNSPSequenceClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertNSPHead(config)
        self.num_labels2 = num_labels2
        self.classifier2 = nn.Linear(config.hidden_size, num_labels2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,output_all_encoded_layers=False)
        seq_relationship_score = self.cls(pooled_output)
        logits2 = self.classifier2(pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.num_labels2), next_sentence_label.view(-1))
            return next_sentence_loss + loss2
        else:
            return seq_relationship_score, logits2


class BertForSequenceClassification(BertPreTrainedModel):
    """文本分类
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForMultipleChoice(BertPreTrainedModel):
    """句子选择：默认计算单选loss，可以用于计算多选loss
    例如：输入，陈述句+备选句1，陈述句+备选句2，陈述句+备选句3，陈述句+备选句4，输出更相关的label
    """
    def __init__(self, config, num_choices):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(BertPreTrainedModel):
    """token 分类
    """
    def __init__(self, config, num_labels):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class BertForTwoSequenceClassification(BertPreTrainedModel):
    """两个文本分类任务
    """
    def __init__(self, config, num_labels1, num_labels2, first_last_avg=False):
        super(BertForTwoSequenceClassification, self).__init__(config)
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.first_last_avg = first_last_avg
        self.output_all_encoded_layers = first_last_avg
        # 为了能够迁移到单分类
        self.classifier = nn.Linear(config.hidden_size, num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, num_labels2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels1=None, labels2=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=self.output_all_encoded_layers,
                                     first_last_avg=self.first_last_avg)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits2 = self.classifier2(pooled_output)

        if labels1 is not None and labels2 is not None:
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(logits.view(-1, self.num_labels1), labels1.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.num_labels2), labels2.view(-1))
            return loss1 + loss2
        else:
            return logits, logits2


class ALBERT(BertPreTrainedModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.embeddings = BertEmbeddings(config)  # embeddings与bert一致
        self.encoder = BertEncoder(config)
        self.pooler = Pooler(config)  # if config.with_pool or config.with_nsp else None
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask=None, output_all_encoded_layers=False):
        if attention_mask is None:
            # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
            # mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)
            attention_mask = (input_ids > 0).unsqueeze(1).unsqueeze(1)  # mask -> [batch_size, 1, 1, seq_len]
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if self.config.with_unilm:
            # TODO add additional mask on attention mask
            pass
        emb_output = self.embeddings(input_ids, token_type_ids)
        encoder_layers = self.encoder(emb_output, attention_mask, output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoder_layers = encoder_layers[-1]
        pooled_output = self.pooler(encoder_layers)

        return encoder_layers, pooled_output


# class NextSentencePrediction(nn.Module):
#     """ x -> pooler: (x[:,0] -> dense -> tanh) -> linear -> log_softmax
#     """
#
#     def __init__(self, config):
#         super().__init__()
#
#         # if not pool, we add a pooler
#         self.pooler = Pooler(config) if not config.with_pool else None
#         self.linear = nn.Linear(config.hidden_size, 2)
#         self.log_softmax = nn.LogSoftmax(dim=-1)
#
#     def forward(self, x):
#         if self.pooler:
#             return self.log_softmax(self.linear(self.pooler(x)))
#         else:
#             # x is pooler
#             return self.log_softmax(self.linear(x))
#
#
# class MaskedLanguageModel(nn.Module):
#     """x -> dense -> act_fn -> norm -> decoder(shared weights) -> log_softmax
#     """
#     def __init__(self, config, embedding_weights):
#         super().__init__()
#
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.act_fn = act2fn.get(config.hidden_act, 'gelu')
#         self.norm = BertLayerNorm(config.hidden_size, config.layer_norm_eps)
#         # The output weights are the same as the input embeddings, but there is
#         # an output-only bias for each token.
#         self.decoder = nn.Linear(embedding_weights.size(1), embedding_weights.size(0), bias=False)
#         self.decoder.weight = embedding_weights
#         self.bias = nn.Parameter(torch.zeros(embedding_weights.size(0)))
#
#         self.log_softmax = nn.LogSoftmax(dim=-1)
#
#     def forward(self, x):
#         x = self.norm(self.act_fn(self.dense(x)))
#         x = self.log_softmax(self.decoder(x) + self.bias)
#         return x



