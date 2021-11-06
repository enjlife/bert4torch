import time
import json
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
from bert_torch import DatasetBase, BertTokenizer, BertForSequenceClassification, Trainer, time_diff
from reference.logger_configuration import get_logger


logger = get_logger()

PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'


def load_dataset(path, max_len=128):
    D = []
    sep_id = tokenizer.convert_tokens_to_ids([SEP])
    label2id = {label: i for i, label in enumerate(config.labels)}
    with open(path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            d = json.loads(line)
            sent, label = d['sentence'], d.get('label', '100')
            tokens = tokenizer.tokenize(sent)
            tokens = [CLS] + tokens
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(token_ids) >= max_len:
                mask = [1] * max_len
                token_ids = token_ids[:max_len-1] + sep_id
            else:
                token_ids += sep_id
                mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
                token_ids += ([0] * (max_len - len(token_ids)))
            type_ids = [0] * max_len
            D.append((token_ids, type_ids, mask, label2id[label]))
    return D


class DataIterator(DatasetBase):
    def __init__(self, data_list, batch_size, device, rand=False):
        super(DataIterator, self).__init__(data_list, batch_size, device, rand)

    def _to_tensor(self, datas):
        token_ids = torch.LongTensor([data[0] for data in datas]).to(self.device)
        type_ids = torch.LongTensor([data[1] for data in datas]).to(self.device)
        mask = torch.ByteTensor([data[2] for data in datas]).to(self.device)
        labels = torch.ByteTensor(data[3] for data in datas).to(self.device)
        return token_ids, type_ids, mask, labels


class Config(object):
    """配置训练参数"""
    def __init__(self):
        self.data_path = '../data/tnews/'  # train.json, valid.json, test.json
        self.labels = ['100', '101', '102', '103', '104', '106', '107', '108', '109', '110', '112', '113', '114', '115', '116']  # 类别名单
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000  # early stopping: 1000 batches
        self.num_classes = len(self.labels)
        self.num_epochs = 10
        self.batch_size = 32
        self.max_len = 128
        self.lr = 5e-4
        self.scheduler = 'CONSTANT_WITH_WARMUP'  # 学习率策略
        self.pretrained_path = '../pretrained_model/bert-base-chinese'
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.01
        self.with_cuda = True
        self.cuda_devices = None
        self.num_warmup_steps = 0.1  # total_batch的一个比例
        self.log_freq = 500
        self.logger = logger


class TNEWSTrainer(Trainer):

    def __init__(self, config, train_iter, dev_iter, model):
        super(TNEWSTrainer, self).__init__(config, train_iter, dev_iter, model)

    def train(self):
        start_time = time.time()
        total_batch = 0
        min_loss = float('inf')
        last_improve = 0
        flag = False  # 用于early-stop
        model.train()
        for epoch in self.num_epochs:
            logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            data_iter = tqdm(enumerate(train_iter),
                             desc="EP_%s:%d" % ('train', epoch),
                             total=len(train_iter),
                             bar_format="{l_bar}{r_bar}")
            for data in data_iter:
                token_ids, type_ids, mask, labels = data
                logits = self.model(token_ids, type_ids, mask)
                model.zero_grad()
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if total_batch % self.log_freq == 0:
                    dev_loss, dev_acc = self.dev()
                    if dev_loss < min_loss:
                        min_loss = dev_loss
                        self.save(epoch)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    t_dif = time_diff(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Val Loss: {2:>5.2},  Val Acc: {3:>6.2%},  Time: {4} {5}'
                    logger.info(msg.format(total_batch, loss, dev_loss, dev_acc, t_dif, improve))
                total_batch += 1
                if total_batch - last_improve > config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    logger.info(f"No optimization for {config.require_improvement} batches, auto-stopping...")
                    flag = True
                    break
            if flag:
                break

    def dev(self, data_iter=None):
        # 计算验证集上的损失和准确率
        model.eval()
        loss_total = 0.0
        acc_total = 0
        item_total = 0
        with torch.no_grad():
            for data in data_iter if data_iter else self.dev_data:
                token_ids, type_ids, mask, labels = data
                logits = model(token_ids, type_ids, mask)
                loss = F.cross_entropy(logits, labels)
                loss_total += loss
                item_total += torch.numel(labels)
                acc_total += torch.sum(logits.argmax(axis=1).eq(labels))
        return loss_total/len(dev_data), acc_total/item_total

    def test(self):
        pass


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)  # dir is OK
    # data
    train_data = load_dataset(config.data_path + 'train.json')
    dev_data = load_dataset(config.data_path + 'valid.json')
    train_iter = DataIterator(train_data, config.batch_size, config.device)
    dev_iter = DataIterator(dev_data, config.batch_size, config.device)

    model = BertForSequenceClassification.from_pretrained(config.pretrained_path)
    trainer = TNEWSTrainer(config, train_iter, dev_iter, model)
    trainer.train()
    trainer.test()

