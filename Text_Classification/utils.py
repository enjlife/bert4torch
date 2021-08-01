# coding: UTF-8
import os
import torch
from tqdm import tqdm
import time
import random
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def load_dataset(path, max_len=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if max_len:
                    if len(token) < max_len:
                        mask = [1] * len(token_ids) + [0] * (max_len - len(token))
                        token_ids += ([0] * (max_len - len(token)))
                    else:
                        mask = [1] * max_len
                        token_ids = token_ids[:max_len]
                        seq_len = max_len
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
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
        x = torch.LongTensor([data[0] for data in datas]).to(self.device)
        y = torch.LongTensor([data[1] for data in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([data[2] for data in datas]).to(self.device)
        mask = torch.LongTensor([data[3] for data in datas]).to(self.device)
        return (x, seq_len, mask), y

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


def process_thucnews(ipath, opath):
    dirs = os.listdir(ipath)
    for label in dirs:
        f_out = open(os.path.join(opath, label), 'w')
        label_dir = os.path.join(ipath, label)
        if not os.path.isdir(label_dir):
            continue
        for file in tqdm(os.listdir(label_dir)):
            title = open(os.path.join(label_dir, file), 'r').readlines()
            if len(title) == 0:
                continue
            f_out.write(title[0])
        f_out.close()


def gen_train_dev_test(ipath, opath):
    label_list = ['财经', '房产', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐']
    data = []
    for idx, label in enumerate(label_list):
        file = os.path.join(ipath, label)
        with open(file, 'r') as f:
            for line in f:
                line = line.strip() + '\t' + str(idx)
                data.append(line)
    random.shuffle(data)
    with open(os.path.join(opath, 'dev.txt'), 'w') as f:
        for line in data[:10000]:
            f.write(line + '\n')
    with open(os.path.join(opath, 'test.txt'), 'w') as f:
        for line in data[10000:20000]:
            f.write(line + '\n')
    with open(os.path.join(opath, 'train.txt'), 'w') as f:
        for line in data[20000:]:
            f.write(line + '\n')
    return


if __name__ == '__main__':
    in_path = '/Users/enjlife/Desktop/data/THUCNews/'
    out_path1 = '/Users/enjlife/Desktop/data/THUCNewsTitles/'
    # process_thucnews(in_path, out_path1)
    out_path2 = '/Users/enjlife/Desktop/data/THUCNewsTextClassification/'
    gen_train_dev_test(out_path1, out_path2)
