# coding: UTF-8
import sys
import os
sys.path.append(os.path.abspath(os.getcwd()+'/..'))
import time
import numpy as np
import torch
from train_eval import train, test
from bert_pytorch import BertConfig, BertForMaskedLM, BertTokenizer
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif


parser = argparse.ArgumentParser(description='Chinese sentiment')
parser.add_argument('--prefix', type=str, default='../')
parser.add_argument('--model_path', type=str, default='pretrained_model/roberta-base-chinese')
parser.add_argument('--save_path', type=str, default='trained_model/model.ckpt')
parser.add_argument('--train_path', type=str, default='data/sentiment/train.data')
parser.add_argument('--dev_path', type=str, default='data/sentiment/dev.data')
parser.add_argument('--test_path', type=str, default='data/sentiment/test.data')
parser.add_argument('--class_path', type=str, default='data/sentiment/class.data')
parser.add_argument('--mode', type=str, default='test', help='train or test')
parser.add_argument('--cuda', type=str, default='cuda:0')

args = parser.parse_args()


class Config(object):
    """配置训练参数"""
    def __init__(self, args):
        self.model_name = 'bert'
        self.mode = args.mode
        self.train_path = args.prefix + args.train_path
        self.dev_path = args.prefix + args.dev_path
        self.test_path = args.prefix + args.test_path
        self.class_list = [x.strip() for x in open(args.prefix + args.class_path).readlines()]  # 类别名单
        self.save_path = args.save_path                  # 模型训练结果
        self.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000                                 # early stopping: 1000 batches
        self.num_classes = len(self.class_list)                         # 类别数: label_num
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 4                                            # mini-batch大小
        self.max_len = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.model_path = args.prefix + args.model_path                               # 预训练模型path
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)  # dir is OK
        # self.hidden_size = 768


if __name__ == '__main__':
    # x = import_module('models.' + model_name)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    config = Config(args)
    print(f'using device:{config.device}')
    start_time = time.time()
    print("Loading data...")
    model = BertForMaskedLM.from_pretrained(config.model_path).to(config.device)
    for pre in ['很理想。', '很理想，']:
        print('-'*10 + pre + '-'*10)
        config.test_prefix = pre
        train_data, dev_data, test_data = build_dataset(config)
        train_iter, dev_iter = None, None
        if args.mode == 'train':
            train_iter = build_iterator(train_data, config)
            dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        # train
        if args.mode == 'train':
            train(config, model, train_iter, dev_iter, test_iter)
        else:
            test(config, model, test_iter)
