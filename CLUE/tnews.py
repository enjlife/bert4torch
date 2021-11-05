import json

import torch.nn
from tqdm import tqdm
from bert_torch import BertTokenizer, BertForSequenceClassification, Trainer
from reference.logger_configuration import get_logger


logger = get_logger()

# TODO build_dataset and trainer

PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'


def load_dataset(path, max_len=128):
    D = []
    sep_id = tokenizer.convert_tokens_to_ids([SEP])
    label2id = {label: i for i, label in enumerate(config.labels)}
    with open(path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            d = json.loads(line)
            sent, label = d['sentence'], d.get('label', '100')
            tokens = tokenizer.tokenize(sent)
            tokens = [CLS] + tokens
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(token_ids) >= max_len:
                mask = [1] * max_len
                token_ids = token_ids[:max_len-1] + sep_id
            else:
                token_ids += sep_id
                mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
                token_ids += ([0] * (max_len - len(token_ids)))
            type_ids = [0] * max_len
            D.append((token_ids, type_ids, mask, label2id[label]))
    return D


class Config(object):
    """配置训练参数"""
    def __init__(self):
        self.data_path = '../data/tnews/'  # train.json, valid.json, test.json
        self.labels = ['100', '101', '102', '103', '104', '106', '107', '108', '109', '110', '112', '113', '114', '115', '116']  # 类别名单
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000  # early stopping: 1000 batches
        self.num_classes = len(self.labels)
        self.num_epochs = 10
        self.batch_size = 32
        self.max_len = 128
        self.lr = 5e-4
        self.scheduler = 'CONSTANT_WITH_WARMUP'
        self.pretrained_path = '../pretrained_model/bert-base-chinese'
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.01
        self.with_cuda = True
        self.cuda_devices = None
        self.log_freq = 100
        self.logger = logger


class TNEWSTrainer(Trainer):

    def __init__(self, config, train_iter, valid_iter):
        super(TNEWSTrainer, self).__init__(config, train_iter, valid_iter)

    def train(self):
        for epoch in self.num_epochs:
            pass

    def valid(self):
        pass

    def test(self):
        pass



if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)  # dir is OK
    model = BertForSequenceClassification.from_pretrained(config.pretrained_path)
