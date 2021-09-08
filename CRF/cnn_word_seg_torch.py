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
            torch.nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(10, 16, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(10, 16, 3, 1, 1, padding_mode='replicate'),
            torch.nn.ReLU()
        )
        self.dense = nn.Linear(hidden_size, 4)
        self.crf = CRF(num_tags=num_labels, batch_first=False)

    def forward(self, x, y, mask):
        hidden_state = self.embedding(x)

        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.conv3(hidden_state)

        hidden_state = self.dense(hidden_state)
        hidden_state = self.crf(hidden_state, y, mask)
        return hidden_state


def build_dataset():
    sents = open('../icwb2-data/training/msr_training.utf8').read().strip().split('\r\n')
    sents = [re.split(' +', s) for s in sents]  # 词之间以两个空格隔开
    sents = [[w for w in s if w] for s in sents]  # 去掉空字符串
    random.shuffle(sents)  # 打乱语料，以便后面划分验证集
    chars, id2char, char2id = build_vocab(sents)


def build_vocab(sents, min_count=5):
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
    return chars, id2char, char2id


if __name__ == '__main__':
    model = CnnWordSeg(4, 128, 10000)
    x = list(model.named_parameters())
    print(model.named_parameters())