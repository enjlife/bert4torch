import os
import torch.nn
from torch import nn
from .crf_torch import CRF
import re
import random


class CnnWordSeg(nn.Module):
    """CNN 分词"""
    def __init__(self, num_labels, vocab_size, hidden_size):
        super(CnnWordSeg, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.conv1 = torch.nn.Sequential(
            # 这里采用重复填充 padding=1填充一层
            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 128, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 128, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.dense = nn.Linear(hidden_size, 4)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, x, y, mask):
        hidden_state = self.embedding(x)  # (batch,seq_len,hidden_size)

        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.conv3(hidden_state)

        hidden_state = self.dense(hidden_state)
        hidden_state = self.crf(hidden_state, y, mask)
        return hidden_state


class DatasetIterater(object):
    def __init__(self, data_list, batch_size, device):
        self.batch_size = batch_size
        self.data_list = data_list
        self.n_batches = len(data_list) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(data_list) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        token_ids = torch.LongTensor([data[0] for data in datas]).to(self.device)
        y = torch.LongTensor([data[1] for data in datas]).to(self.device)
        token_type_ids = torch.LongTensor([data[2] for data in datas]).to(self.device)
        mask = torch.LongTensor([data[3] for data in datas]).to(self.device)
        return (token_ids, token_type_ids, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.data_list[self.index * self.batch_size: len(self.data_list)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.data_list[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_dataset(path):
    sents = open(path, 'r').read().strip().split('\r\n')
    sents = [re.split(' +', s) for s in sents]  # 词之间以两个空格隔开
    sents = [[w for w in s if w] for s in sents]  # 去掉空字符串
    random.shuffle(sents)  # 打乱语料，以便后面划分验证集
    id2char, char2id = build_vocab(sents)
    trains, valids = sents[:-5000], sents[-5000:]


def to_id(data, char2id):
    datasets = []
    for s in data:
        x, y = [], []
        for w in s:
            if not all(c in char2id for c in w):
                continue
            x.extend([char2id[c] for c in w])
            if len(w) == 1:
                y.append(0)
            elif len(w) == 2:
                y.extend([1, 3])
            else:
                y.append()


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def build_vocab(sents, min_count=2):
    chars = {}
    for s in sents:
        for c in ''.join(s):
            if c in chars:
                chars[c] += 1
            else:
                chars[c] = 1
    chars = {i: j for i, j in chars.items() if j >= min_count}
    id2char = {i: j for i, j in enumerate(chars.keys())}
    char2id = {j: i for i, j in id2char.items()}
    return id2char, char2id


if __name__ == '__main__':
    path = '../icwb2-data/training/msr_training.utf8'
    model = CnnWordSeg(4, 128, 10000)
    x = list(model.named_parameters())
    print(model.named_parameters())