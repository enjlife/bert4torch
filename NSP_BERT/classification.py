import sys
import time
import json
import numpy as np
import torch.nn
from tqdm import tqdm
sys.path.append('../')
from bert_torch import DatasetBase, BertTokenizer, BertForNextSentencePrediction, Trainer, time_diff, sequence_padding
from reference.logger_configuration import _get_library_root_logger
from sklearn import metrics

"""Zero-shot eprstmt电商评论数据集分类 测试集acc：86.8%
"""
logger = _get_library_root_logger()

PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'


def load_dataset(path, patterns, max_len=256):
    """返回与模板数量相同的数据集"""
    D = [[] for _ in patterns]
    labels = []
    patterns = [tokenizer.tokenize(p) for p in patterns]
    second_text_len = max_len - 3 - max([len(p) for p in patterns])
    label2id = {label: i for i, label in enumerate(config.labels)}
    fr = open(path, 'r', encoding='utf8')
    for line in tqdm(fr):
        d = json.loads(line)
        sent, label = d['sentence'], label2id[d['label']]
        tokens = tokenizer.tokenize(sent)
        tokens = tokens[:second_text_len]
        for i, p in enumerate(patterns):
            token_ids = tokenizer.convert_tokens_to_ids([CLS] + p + [SEP] + tokens + [SEP])
            seg_ids = [0]*(len(p)+2) + [1]*(len(tokens)+1)
            D[i].append([token_ids, seg_ids])
        labels.append(label)
    fr.close()
    return D, labels


class DataIterator(DatasetBase):
    def __init__(self, data_list, batch_size, device, rand=False):
        super(DataIterator, self).__init__(data_list, batch_size, device, rand)

    def _to_tensor(self, datas):
        token_ids = [data[0] for data in datas]
        seg_ids = [data[1] for data in datas]
        token_ids = sequence_padding(token_ids)
        seg_ids = sequence_padding(seg_ids)
        token_ids = torch.LongTensor(token_ids).to(self.device)
        seg_ids = torch.LongTensor(seg_ids).to(self.device)
        return token_ids, seg_ids


class Config(object):
    """配置训练参数"""
    def __init__(self):
        self.data_path = '../data/eprstmt/'  # train.json, valid.json, test.json
        self.labels = ['Positive', 'Negative']  # 类别名单
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 10000  # early stopping: 1000 batches
        self.num_classes = len(self.labels)
        self.num_epochs = 10
        self.batch_size = 16
        self.max_len = 128
        self.lr = 5e-4
        self.scheduler = 'CONSTANT'  # 'CONSTANT_WITH_WARMUP'  学习率策略
        self.pretrained_path = '../pretrained_model/uer-bert-base'
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.01
        self.with_cuda = True
        self.cuda_devices = None
        self.num_warmup_steps = 0.1  # total_batch的一个比例
        self.log_freq = 1000
        self.logger = logger
        self.save_path = 'trained.model'


class CLSTrainer(Trainer):
    def __init__(self, config, model):
        self.data_path = config.data_path
        self.labels = config.labels
        super(CLSTrainer, self).__init__(config, model)

    def test(self, data_iters, labels):
        self.model.eval()
        y_preds = [[] for _ in range(len(data_iters))]
        with torch.no_grad():
            for i, data_iter in enumerate(data_iters):
                for (token_ids, seg_ids) in data_iter:
                    logits = self.model(token_ids, seg_ids).cpu().numpy()
                    y_preds[i].extend(logits[:, 0].tolist())
        y_preds = np.argmax(y_preds, axis=0)
        confusion_matrix = metrics.confusion_matrix(np.array(labels), y_preds)
        logger.info("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
        acc = metrics.accuracy_score(np.array(labels), y_preds, normalize=True, sample_weight=None)
        logger.info("Acc.:\t{:.4f}".format(acc), flush=True)


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)  # dir is OK
    # data
    patterns = ['这次买的东西很好', '这次买的东西很差']
    test_data, labels = load_dataset(config.data_path + 'test_public.json', patterns)
    test_iters = [DataIterator(data, config.batch_size, config.device) for data in test_data]

    model = BertForNextSentencePrediction.from_pretrained(config.pretrained_path)
    trainer = CLSTrainer(config, model)
    trainer.test(test_iters, labels)

