import unittest
from bert4keras.models import build_transformer_model
# from bert_pytorch import BERT, BertConfig
# from transformers import models
from bert4keras import models
#
# class BERTVocabTestCase(unittest.TestCase):
#     pass
#
# models.bert

# bert = BERT.from_pretrained(r'D:\\BERT-pytorch-master\\pretrained_model\\bert-base-chinese')

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)
bert.model.layers[-2].get_output_at(-1)