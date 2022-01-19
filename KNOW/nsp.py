import random
import sys
import time
import json
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append('../')
from bert_torch import DatasetBase, BertTokenizer, BertForNextSentencePrediction, Trainer, time_diff, \
                        sequence_padding, set_seed, get_logger

logger = get_logger()


def load_dataset(path, pattern, max_len=128):
    # 微博文本预处理后最大140，所以这里不需要max_len
    PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'
    D = []
    label2id = {label: i for i, label in enumerate(config.labels)}
    fr = open(path, 'r', encoding='utf8')
    for line in tqdm(fr):
        content, space, label = line.strip().split('\t')[1:]  # mid,content,space,label
        pattern_tokens = tokenizer.tokenize(pattern.format(space))
        tokens = tokenizer.tokenize(content)
        # seg_ids
        type_ids = [0] * (len(pattern_tokens) + 2) + [1] * (len(tokens) + 1)
        tokens = [CLS] + pattern_tokens + [SEP] + tokens + [SEP]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        D.append((token_ids, type_ids, label2id[label]))
        # random.shuffle(D)
    fr.close()
    return D


class DataIterator(DatasetBase):

    def __init__(self, data_list, batch_size, rand=False):
        super(DataIterator, self).__init__(data_list, batch_size, rand)

    def _to_tensor(self, datas):
        token_ids = [data[0] for data in datas]
        type_ids = [data[1] for data in datas]
        token_ids = sequence_padding(token_ids)
        type_ids = sequence_padding(type_ids)
        token_ids = torch.LongTensor(token_ids)
        type_ids = torch.LongTensor(type_ids)
        labels = torch.LongTensor([data[2] for data in datas])
        return token_ids, type_ids, labels


class Config(object):
    """配置训练参数"""
    def __init__(self):
        self.pretrained_path = '../../pretrained_model/bert-base-chinese'
        self.data_path = '../data/space/'  # train.json, valid.json, test.json
        self.labels = ['1', '0']  # 类别名单 1->0 0->1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 10000  # early stopping: 1000 batches
        self.num_classes = len(self.labels)
        self.num_epochs = 3
        self.num_batches = 0
        self.batch_size = 16
        self.max_len = 128
        self.lr = 2e-5  # 5e-4
        self.scheduler = 'CONSTANT'  # 学习率策略'CONSTANT','CONSTANT_WITH_WARMUP'
        self.max_grad_norm = 1.0  # 梯度裁剪
        self.gradient_accumulation_steps = 1
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.01
        self.cuda_devices = None
        self.num_warmup_steps_ratio = 0.1  # total_batch的一个比例
        self.log_mode = 'epoch_end'
        self.log_freq = 25
        self.save_path = 'trained2.model'
        self.with_drop = False  # 分类的全连接层前是否dropout


class SPACETrainer(Trainer):
    def __init__(self, config, model):
        self.data_path = config.data_path
        self.labels = config.labels
        super(SPACETrainer, self).__init__(config, model)

    def train(self, train_iter=None, dev_iter=None):
        start_time = time.time()
        step = 0
        min_loss = float('inf')
        last_improve = 0
        flag = False  # 用于early-stop
        self.model.train()
        self.model.zero_grad()
        for epoch in range(self.num_epochs):
            logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            data_iter = tqdm(train_iter, desc="EP_%s:%d" % ('train', epoch))  # bar_format="{l_bar}{r_bar}"
            for (token_ids, type_ids, labels) in data_iter:
                token_ids, type_ids, labels = token_ids.to(self.device), type_ids.to(self.device), labels.to(self.device)
                logits = self.model(token_ids, type_ids)
                loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                    self.scheduler.step()
                if step % self.log_freq == 0:
                    dev_loss, dev_acc = self.dev(dev_iter=dev_iter)
                    if dev_loss < min_loss:
                        min_loss = dev_loss
                        self.save_model(epoch)
                        improve = '*'
                        last_improve = step
                    else:
                        improve = ''
                    t_dif = time_diff(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Val Loss: {2:>5.2},  Val Acc: {3:>6.2%},  Time: {4} {5}'
                    logger.info(msg.format(step, loss, dev_loss, dev_acc, t_dif, improve))
                step += 1
                # early_stopping
                if step - last_improve > config.require_improvement:
                    logger.info(f"No optimization for {config.require_improvement} batches, auto-stopping...")
                    flag = True
                    break
            if flag:
                break

    def dev(self, dev_iter=None, train_iter=None):
        # 计算验证集上的损失和准确率
        self.model.eval()
        loss_total = 0.0
        acc_total = 0
        item_total = 0
        with torch.no_grad():
            for (token_ids, type_ids, labels) in dev_iter:
                token_ids, type_ids, labels = token_ids.to(self.device), type_ids.to(self.device), labels.to(
                    self.device)
                logits = self.model(token_ids, type_ids)
                loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
                loss_total += loss.item()
                item_total += torch.numel(labels)
                acc_total += torch.sum(logits.argmax(axis=1).eq(labels)).item()
        self.model.train()
        return loss_total/item_total, acc_total/item_total
        # return loss_total/len(dev_data), acc_total/item_total

    def test(self, test_iter, model_path=None):
        if not model_path:
            model_path = self.save_path
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        res = []
        with torch.no_grad():
            for (token_ids, type_ids, _) in test_iter:
                token_ids, type_ids = token_ids.to(self.device), type_ids.to(self.device)
                logits = self.model(token_ids, type_ids)
                logits = logits.argmax(axis=1)
                logits = logits.cpu().numpy().tolist()
                res.extend(logits)
        self.model.train()
        return res


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)  # dir is OK
    # train_data = load_dataset(config.data_path + 'train.json')
    patterns = ['{}', '博主在介绍{}', '博主介绍的地方是{}。', '博主所在的地方是{}。', '{}是博客涉及的地方', '该博客的地域标签是{}']

    mode = 'dev'
    if mode == 'train':
        data = load_dataset(config.data_path + 'train.txt', patterns[-1])
        train_iter, dev_iter = DataIterator(data[:800], config.batch_size), DataIterator(data[800:], config.batch_size)
        config.num_batches = len(train_iter)
        set_seed(42)
        cls_model = BertForNextSentencePrediction.from_pretrained(config.pretrained_path)
        # model.load_state_dict(torch.load('trained2.model'))
        trainer = SPACETrainer(config, cls_model)
        trainer.train(train_iter, dev_iter)
    elif mode == 'dev':
        dev_datas = []
        for p in patterns:
            dev_datas.append(load_dataset(config.data_path + 'train.txt', p))
        # config.train_iter = DataIterator(train_data, config.batch_size)
        dev_iters = [DataIterator(dev_data, config.batch_size) for dev_data in dev_datas]
        cls_model = BertForNextSentencePrediction.from_pretrained(config.pretrained_path)
        trainer = SPACETrainer(config, cls_model)
        for i, dev_iter in enumerate(dev_iters):
            loss, acc = trainer.dev(dev_iter)
            logger.info('pattern: {}\tloss: {:>5.2}\tacc: {:>6.3f}'.format(patterns[i], loss, acc))

        trainer.model.load_state_dict(torch.load(config.save_path))
        loss, acc = trainer.dev(dev_iters[-1])
        logger.info('pattern: {}\tloss: {:>5.2}\tacc: {:>6.3}'.format(patterns[-1], loss, acc))

    else:
        test_data = load_dataset(config.data_path + 'test.txt', patterns[-1])
        test_iter = DataIterator(test_data, config.batch_size)
        cls_model = BertForNextSentencePrediction.from_pretrained(config.pretrained_path)
        trainer = SPACETrainer(config, cls_model)
        res = trainer.test(test_iter, 'trained2.model')
        id2label = {0: '1', 1: '0'}
        with open(config.data_path + 'test.txt', 'r') as f:
            l = [line.strip() for line in f.readlines()]
        fw = open(config.data_path + 'predict.txt', 'w', encoding='utf8')
        for label, line in zip(res, l):
            fw.write('{}\t{}\n'.format(line, id2label[label]))



