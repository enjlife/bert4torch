import time
import json
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
from bert_torch import DatasetBase, BertTokenizer, BertForSequenceClassification, Trainer, time_diff, \
                        sequence_padding, set_seed, get_logger

# iflytek 分类任务

logger = get_logger()


def load_dataset(path, max_len=128):
    PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'
    D = []
    fr = open(path, 'r', encoding='utf8')
    for line in tqdm(fr):
        d = json.loads(line)
        sent, label = d['sentence'], d.get('label', '0')
        tokens = tokenizer.tokenize(sent)
        tokens = [CLS] + tokens[:max_len-2] + [SEP]  # tokens最大为max_len-2
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        type_ids = [0] * len(token_ids)
        D.append((token_ids, type_ids, int(label)))
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
        self.pretrained_path = '../pretrained_model/uer-bert-base'
        self.data_path = '../data/iflytek/'  # train.json, valid.json, test.json
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 10000  # early stopping: 1000 batches
        self.num_classes = 119
        self.num_epochs = 8
        self.num_batches = 0
        self.batch_size = 16
        self.max_len = 128
        self.lr = 2e-5  # 5e-4
        self.scheduler = 'CONSTANT'  # 学习率策略 'CONSTANT','CONSTANT_WITH_WARMUP'
        self.max_grad_norm = 1.0  # 梯度裁剪
        self.gradient_accumulation_steps = 2
        self.betas = (0.9, 0.999)
        self.weight_decay = 0.01
        self.with_cuda = True
        self.cuda_devices = None
        self.num_warmup_steps_ratio = 0.1  # total_batch的一个比例
        self.log_freq = 770
        self.save_path = 'trained_iflytek.model'
        self.with_drop = False  # 分类的全连接层前是否dropout


class IFLYTEKTrainer(Trainer):
    def __init__(self, config, model):
        self.data_path = config.data_path
        self.num_classes = config.num_classes

        super(IFLYTEKTrainer, self).__init__(config, model)

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
                loss = F.cross_entropy(logits.view(-1, self.num_classes), labels.view(-1))
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

    def dev(self, train_iter=None, dev_iter=None):
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
                loss = F.cross_entropy(logits.view(-1, self.num_classes), labels.view(-1))
                loss_total += loss.item()
                item_total += torch.numel(labels)
                acc_total += torch.sum(logits.argmax(axis=1).eq(labels)).item()
        self.model.train()
        return loss_total/len(dev_data), acc_total/item_total

    def test(self):
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        test_data = load_dataset(self.data_path + 'test.json')
        test_iter = DataIterator(test_data, config.batch_size)
        res = []
        with torch.no_grad():
            for (token_ids, type_ids, _) in test_iter:
                token_ids, type_ids = token_ids.to(self.device), type_ids.to(self.device)
                logits = self.model(token_ids, type_ids)
                logits = logits.argmax(axis=1)
                logits = logits.cpu().numpy().tolist()
                res.extend(logits)
        fw = open(self.data_path + 'iflytek_predict.json', 'w')
        for i, r in enumerate(res):
            l = json.dumps({'id': str(i), 'label': str(r)})
            fw.write(l + '\n')
        fw.close()
        self.model.train()


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_path)  # dir is OK
    # data
    train_data = load_dataset(config.data_path + 'train.json')
    dev_data = load_dataset(config.data_path + 'dev.json')
    train_iter = DataIterator(train_data, config.batch_size)
    dev_iter = DataIterator(dev_data, config.batch_size)
    config.num_batches = len(train_iter)

    set_seed(42)
    cls_model = BertForSequenceClassification.from_pretrained(config.pretrained_path, config.num_classes)
    # model.load_state_dict(torch.load('trained2.model'))
    trainer = IFLYTEKTrainer(config, cls_model)
    trainer.train(train_iter, dev_iter)
    trainer.test()
    # loss, acc = trainer.dev()
    # print('loss: %f, acc: %f' % (loss, acc))
    # trainer.test()

