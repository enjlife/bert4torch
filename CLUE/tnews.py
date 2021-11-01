import os
import torch.nn
from torch import nn
import re
import random
import time
from torch.optim import Adam
import torch.nn.functional as F
from datetime import timedelta
from tqdm import tqdm
from utils import DatasetBase
from bert_pytorch import BertTokenizer, BertForSequenceClassification
from ..utils.logger_configuration import get_logger


logger = get_logger()

# TODO build_dataset

PAD, CLS, MASK = '[PAD]', '[CLS]', '[MASK]'

def build_dataset(config):
    def load_dataset(path, max_len=128):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                content = prefix + content
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                token[mask_idx] = MASK  # 'CLS MASK ...'
                # seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if max_len:
                    if len(token) < max_len:
                        mask = [1] * len(token_ids) + [0] * (max_len - len(token))
                        token_ids += ([0] * (max_len - len(token)))
                    else:
                        mask = [1] * max_len
                        token_ids = token_ids[:max_len]
                        # seq_len = max_len
                token_type_ids = [0] * max_len
                contents.append((token_ids, int(label), token_type_ids, mask, label_ids[int(label)]))
        return contents
    train = load_dataset(config.train_path, config.test_prefix, config.max_len)
    dev = load_dataset(config.dev_path, config.test_prefix, config.max_len)
    return train, dev


class Config(object):
    """配置训练参数"""
    def __init__(self):
        self.train_path = '../data/tnews/train.json'
        self.dev_path = '../data/tnews/valid.json'
        self.test_path = '../data/tnews.test.json'
        self.labels = ['100', '101', '102', '103', '104', '106', '107', '108', '109', '110', '112', '113', '114', '115', '116']  # 类别名单
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000  # early stopping: 1000 batches
        self.num_classes = len(self.labels)
        self.num_epochs = 10
        self.batch_size = 32
        self.max_len = 128
        self.lr = 5e-4
        self.pretrained_path = '../pretrained_model/bert-base-chinese'
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.01
        self.with_cuda = True
        self.cuda_devices = None
        self.log_freq = 100
        self.logger = logger


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)  # dir is OK
    model = BertForSequenceClassification.from_pretrained(config.pretrained_path).to(config.device)