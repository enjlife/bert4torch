import os
import torch.nn
from torch import nn
from crf_torch import CRF
import re
import random
import time
from torch.optim import Adam
import torch.nn.functional as F
from datetime import timedelta

# TODO 数据准备 to_id()
# TODO 数据准备 打乱数据
# TODO batch的准召
# TODO build Train


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
        hidden_state = hidden_state.permute(0, 2, 1)  # 一维卷积是在length维度
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.conv3(hidden_state)
        hidden_state = hidden_state.permute(0, 2, 1)

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
        max_len = max([len(data[0]) for data in datas])
        x = torch.LongTensor([data[0] + [0]*(max_len-len(data[0])) for data in datas]).to(self.device)
        y = torch.LongTensor([data[1] + [0]*(max_len-len(data[0])) for data in datas]).to(self.device)
        mask = torch.ByteTensor([data[2] + [0]*(max_len-len(data[0])) for data in datas]).to(self.device)
        return x, y, mask

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
    sents = open(path, 'r', encoding='utf8').read().strip().split('\n')
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
        id2char = {i+1: j for i, j in enumerate(chars.keys())}
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
            if x:
                datasets.append((x, y, [1]*len(x)))  # x,y,mask
        return datasets
    data = to_id()
    trains, valids = data[:-5000], data[-5000:]
    return trains, valids, id2char, char2id


class Train:
    def __init__(self, model, train_iter, dev_iter, config):
        self.model = model
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.config = config

    def train(self):
        start_time = time.time()
        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')  # dev 最小loss
        for epoch in range(self.config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, self.config.num_epochs))
            for i, (x, y, mask) in enumerate(self.train_iter):
                self.model.zero_grad()
                loss = self.model(x, y, mask)
                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    y_pre = self.model(x, y, mask, test=True)
                    y_true = y.cpu().numpy().tolist()
                    mask = mask.cpu().numpy().sum(axis=1).tolist()
                    train_acc, rec = self.cal_acc(y_pre, y_true, mask)
                    dev_loss, dev_acc, dev_rec = self.evaluate()
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Rec: {3:>6.2%},  Val Loss: {4:>5.2},  Val Acc: {5:>6.2%},  Time: {6} {7}'
                    print(msg.format(total_batch, loss.item(), train_acc, rec, dev_loss, dev_acc, time_dif, improve))
                    model.train()
                total_batch += 1

    def evaluate(self):
        self.model.eval()
        loss_total = 0.0
        acc_total = 0.0
        rec_total = 0.0
        n = 0
        with torch.no_grad():
            for x, y, mask in self.dev_iter:
                loss = self.model(x, y, mask)
                loss_total += loss.item()
                y_pre = self.model(x, y, mask, test=True)
                y_true = y.cpu().numpy().tolist()
                mask = mask.cpu().numpy().sum(axis=1).tolist()
                acc, rec = self.cal_acc(y_pre, y_true, mask)
                acc_total += acc
                rec_total += rec
                n += 1
        return loss_total/n, acc_total/n, rec_total/n

    def cal_acc(self, y_pre, y_true, mask):
        n = len(y_pre)
        acc, rec = 0.0, 0.0
        for i in range(n):
            length = mask[i]
            tp = y_pre[i][:length]
            tt = y_true[i][:length]
            tt = set([i*2 + x for i, x in enumerate(tt) if x == 0 or x == 1])
            tp = set([i*2 + x for i, x in enumerate(tp) if x == 0 or x == 1])
            acc += len(tt & tp) / (len(tp)+1)
            rec += len(tt & tp) / (len(tt)+1)
        return acc/n, rec/n


class Config:
    def __init__(self):
        self.lr = 1e-3
        self.num_epochs = 10
        self.batch_size = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = 4
        self.hidden_size = 128
        self.path = '../data/icwb2/msr_training.utf8'
        self.num_labels = 4
        self.vocab_size = 0
        self.save_path = 'model.ckpt'


if __name__ == '__main__':
    config = Config()
    train_data, valid_data, id2char, char2id = build_dataset(config.path)
    config.vocab_size = len(id2char) + 1
    train_iter = DatasetIterater(train_data, config.batch_size, config.device)
    valid_iter = DatasetIterater(valid_data, config.batch_size, config.device)

    model = CnnWordSeg(config).cuda(0)
    train = Train(model, train_iter, valid_iter, config)
    train.train()