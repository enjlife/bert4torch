import unittest
from bert_pytorch import BERT, BertConfig
from transformers import models
from bert4keras import models
#
# class BERTVocabTestCase(unittest.TestCase):
#     pass
#
# models.bert

bert = BERT.from_pretrained(r'D:\\BERT-pytorch-master\\pretrained_model\\bert-base-chinese')

print(bert)