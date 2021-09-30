import os
import torch.nn
from torch import nn
from .crf_torch import CRF
import re
import random
import time
from torch.optim import Adam
import torch.nn.functional as F
from datetime import timedelta

# TODO 数据准备 to_id()


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class CnnWordSeg(nn.Module):
    """CNN 分词"""
    def __init__(self, config):
        super(CnnWordSeg, self).__init__()
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        num_labels = config.num_labels
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.conv1 = torch.nn.Sequential(
            # 这里采用重复填充 padding=1填充一层
            torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                            kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_size, hidden_size, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_size, hidden_size, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.dense = nn.Linear(hidden_size, 4)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, x, y, mask, test=False):
        hidden_state = self.embedding(x)  # (batch,seq_len,hidden_size)

        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.conv3(hidden_state)

        hidden_state = self.dense(hidden_state)
        if not test:
            hidden_state = self.crf(hidden_state, y, mask)
        else:
            hidden_state = self.crf.decode(hidden_state, mask)
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
        x = torch.LongTensor([data[0] for data in datas]).to(self.device)
        y = torch.LongTensor([data[1] for data in datas]).to(self.device)
        mask = torch.LongTensor([data[2] for data in datas]).to(self.device)
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


def build_dataset(path, max_len=32):
    sents = open(path, 'r').read().strip().split('\r\n')
    sents = [re.split(' +', s) for s in sents]  # 词之间以两个空格隔开
    sents = [[w for w in s if w] for s in sents]  # 去掉空字符串
    random.shuffle(sents)  # 打乱语料，以便后面划分验证集

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
    id2char, char2id = build_vocab(sents)

    def to_id():
        datasets = []
        for s in sents:
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
                    y.extend([1] + [2] * (len(w) - 2) + [3])
            datasets.append((x, y))
        return datasets
    data = to_id()
    trains, valids = data[:-5000], data[-5000:]
    return trains, valids, id2char, char2id


def train(model, train_iter, dev_iter, config):
    start_time = time.time()
    model.train()
    optimizer = Adam(model.parameters(), lr=config.lr)
    total_batch = 0                 # 记录进行到多少batch
    dev_best_loss = float('inf')    # dev 最小loss
    last_improve = 0                # 记录上次验证集loss下降的batch数
    flag = False                    # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (x, y, mask) in enumerate(train_iter):
            model.zero_grad()
            loss = model(x, y, mask)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                predic = model(x, y, mask, test=True)
                true = y.data.cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print(f"No optimization for {config.require_improvement} batches, auto-stopping...")
                flag = True
                break
        if flag:
            break


class Config:
    def __init__(self):
        self.lr = 1e-3
        self.num_epochs = 10
        self.batch_size = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = 4
        self.hidden_size = 128
        self.path = '../icwb2-data/training/msr_training.utf8'
        self.num_labels = 4
        self.vocab_size = 0


if __name__ == '__main__':
    config = Config()
    train_data, valid_data, id2char, char2id = build_dataset(config.path)
    config.vocab_size = len(id2char)
    train_iter = DatasetIterater(train_data, config.batch_size, config.device)
    valid_iter = DatasetIterater(valid_data, config.batch_size, config.device)

    model = CnnWordSeg(config)
    train(model, train_iter, valid_iter, config)