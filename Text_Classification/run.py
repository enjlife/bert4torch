# coding: UTF-8
import time
import torch
from train_eval import train
from bert_pytorch import BertConfig, BertForSequenceClassification, BertTokenizer
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif


parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


class Config(object):
    """配置训练参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = '../data/' + dataset + '/train.txt'
        self.dev_path = '../data/' + dataset + '/dev.txt'
        self.test_path = '../data/' + dataset + '/test.txt'
        self.class_list = [x.strip() for x in open('../data/' + dataset + 'class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000                                 # early stopping: 1000 batches
        self.num_classes = len(self.class_list)                         # 类别数: label_num
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.max_len = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.model_path = '../pretrained_model/bert-base-chinese'        # 预训练模型path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # dir is OK
        # self.hidden_size = 768


if __name__ == '__main__':
    dataset = 'THUCNewsTextClassification'  # 数据集

    model_name = args.model  # bert
    # x = import_module('models.' + model_name)
    config = Config(dataset)

    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = BertForSequenceClassification.from_pretrained(config.model_path, config.num_classes).to(config.device)
    # next(bert.parameters()).is_cuda  -> check if model is on cuda
    train(config, model, train_iter, dev_iter, test_iter)
