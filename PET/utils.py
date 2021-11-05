# coding: UTF-8
import os
import torch
from tqdm import tqdm
import time
import random
from datetime import timedelta

PAD, CLS, MASK = '[PAD]', '[CLS]', '[MASK]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def load_dataset(path, prefix='很理想。', max_len=128):
        mask_idx = 1
        pos_id = config.tokenizer.vocab['很']
        neg_id = config.tokenizer.vocab['不']
        label_ids = [neg_id, pos_id]
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
                        token_ids += ([0] * (max_len - len(token_ids)))
                    else:
                        mask = [1] * max_len
                        token_ids = token_ids[:max_len]
                        # seq_len = max_len
                token_type_ids = [0] * max_len
                contents.append((token_ids, int(label), token_type_ids, mask, label_ids[int(label)]))
        return contents
    train, dev = None, None
    test = load_dataset(config.test_path, config.test_prefix, config.max_len)
    if config.mode == 'train':
        train = load_dataset(config.train_path, config.test_prefix, config.max_len)
        dev = load_dataset(config.dev_path, config.test_prefix, config.max_len)
    return train, dev, test


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
        label_ids = torch.LongTensor([data[4] for data in datas]).to(self.device)
        return (token_ids, token_type_ids, mask), (y, label_ids)

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


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    pass
