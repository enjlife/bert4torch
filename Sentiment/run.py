# coding: UTF-8
import time
import torch
from train_eval import train, test
from bert_pytorch import BertConfig, BertForMaskedLM, BertTokenizer
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model_path', type=str, default='../pretrained_model/bert-base-chinese', help='pretrained model path')
parser.add_argument('--train_path', type=str, default='D:\\sentiment_data\\sentiment\\train')
parser.add_argument('--dev_path', type=str, default='D:\\sentiment_data\\sentiment\\dev')
parser.add_argument('--test_path', type=str, default='D:\\sentiment_data\\sentiment\\test')
parser.add_argument('--class_path', type=str, default='D:\\sentiment_data\\sentiment\\class.txt')
parser.add_argument('--mode', type=str, default='test', help='train or test')

args = parser.parse_args()


class Config(object):
    """配置训练参数"""
    def __init__(self, args):
        self.model_name = 'bert'
        self.mode = args.mode
        self.train_path = args.train_path
        self.dev_path = args.dev_path
        self.test_path = args.test_path
        self.class_list = [x.strip() for x in open(args.class_path).readlines()]  # 类别名单
        self.save_path = '.trained_model' + '/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000                                 # early stopping: 1000 batches
        self.num_classes = len(self.class_list)                         # 类别数: label_num
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.max_len = 128                                              # 每句话处理成的长度(短填长切)
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
    if args.mode == 'train':
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    model = BertForMaskedLM.from_pretrained(config.model_path).to(config.device)
    # train
    if args.mode == 'train':
        train(config, model, train_iter, dev_iter, test_iter)
    else:
        test(config, model, test_iter)
