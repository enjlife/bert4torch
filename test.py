import unittest
from bert4keras.models import build_transformer_model
# from bert_pytorch import BERT, BertConfig
# from transformers import models
from bert4keras import models
from bert_pytorch import BertModel
#
# class BERTVocabTestCase(unittest.TestCase):
#     pass
#
# models.bert

bert = BertModel.from_pretrained('bert-base-chinese', **{'cache_dir': r'pretrained_model/bert-base-chinese'})


# bert = build_transformer_model(
#     config_path=config_path,
#     checkpoint_path=checkpoint_path,
#     return_keras_model=False,
# )
# bert.model.layers[-2].get_output_at(-1)
