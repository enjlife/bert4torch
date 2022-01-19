import random
import sys
import time
import argparse
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append('../')
from bert_torch import DatasetBase, BertTokenizer, BertForSequenceClassification, Trainer, time_diff, \
                        sequence_padding, set_seed, get_logger

logger = get_logger()


def load_dataset(path, max_len=128, mode='train'):
    # 微博文本预处理后最大140，所以这里不需要max_len
    PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'
    D = []
    fr = open(path, 'r', encoding='utf8')
    for line in tqdm(fr):
        line = line.strip('\n').split('\t')
        if len(line) != 4:
            continue
        # mid,label,tag,content
        content, label = line[-1], int(line[1])
        tokens = tokenizer.tokenize(content)
        # seg_ids
        type_ids = [0] * (len(tokens) + 2)
        tokens = [CLS] + tokens + [SEP]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        D.append((token_ids, type_ids, label))
    if mode == 'train':
        random.shuffle(D)
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


class KNOWTrainer(Trainer):
    def __init__(self, config, model):
        self.labels = config.labels
        super(KNOWTrainer, self).__init__(config, model)

    def train(self, train_iter=None, dev_iter=None, test_iter=None):
        start_time = time.time()
        step = 0
        train_loss = 0.0
        min_loss, max_f1 = float('inf'), 0.0
        last_improve = 0
        flag = False  # 用于early-stop
        self.model.train()
        self.model.zero_grad()
        for epoch in range(self.num_epochs):
            logger.info('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))
            tq_iter = tqdm(train_it, desc="EP_%s:%d" % ('train', epoch + 1))  # bar_format="{l_bar}{r_bar}"
            for (token_ids, type_ids, labels) in tq_iter:
                token_ids, type_ids, labels = token_ids.to(self.device), type_ids.to(self.device), labels.to(self.device)
                logits = self.model(token_ids, type_ids)
                loss = self.loss_fct(logits.view(-1, 2), labels.view(-1))
                train_loss += loss.item()
                # tq_iter.set_postfix(loss=loss.item())
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                    self.scheduler.step()
                if step % self.log_freq == 0:
                    train_loss /= self.log_freq
                    dev_loss, p1, r1, f1 = self.dev(dev_iter=dev_iter)
                    # if dev_loss < min_loss:
                    if f1 > max_f1:
                        # min_loss = dev_loss
                        max_f1 = f1
                        self.save_model(epoch)
                        improve = '^^^'
                        last_improve = step
                    else:
                        improve = ''
                    t_dif = time_diff(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Val Loss: {2:>5.2}, f1: {3:>5.3}, p1: {4:>6.2%}, r1: {5:>6.2%}, Time: {6} {7}'
                    logger.info(msg.format(step, train_loss, dev_loss, f1, p1, r1, t_dif, improve))
                    train_loss = 0.0
                    _, _ = self.test(test_iter)
                step += 1
                # early_stopping
                if step - last_improve > self.require_improvement:
                    logger.info(f"No optimization for {self.require_improvement} batches, auto-stopping...")
                    flag = True
                    break

            if flag:
                break

    def dev(self, dev_iter=None, train_iter=None):
        """计算验证集上的损失和准确率"""
        self.model.eval()
        confusion_matrix = torch.zeros(2, 2)
        loss_total = 0.0
        with torch.no_grad():
            for (token_ids, type_ids, labels) in dev_iter:
                token_ids, type_ids, labels = token_ids.to(self.device), type_ids.to(self.device), labels.to(self.device)
                logits = self.model(token_ids, type_ids)
                loss = self.loss_fct(logits.view(-1, 2), labels.view(-1))
                loss_total += loss.item()
                logits = logits.argmax(axis=1)
                for t, p in zip(labels.view(-1), logits.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        self.model.train()
        p1 = confusion_matrix[1, 1] / confusion_matrix[:, 1].sum()
        r1 = confusion_matrix[1, 1] / confusion_matrix[1, :].sum()
        f1 = 2 * p1 * r1 / (p1 + r1)
        return loss_total/len(dev_iter), p1, r1, f1

    def test(self, test_iter, model_path=None, reload=False):
        if not model_path:
            model_path = self.save_path
        if reload:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        confusion_matrix = torch.zeros(2, 2)
        res, score = [], []
        with torch.no_grad():
            for (token_ids, type_ids, labels) in test_iter:
                token_ids, type_ids, labels = token_ids.to(self.device), type_ids.to(self.device), labels.to(self.device)
                logits = self.model(token_ids, type_ids)
                logits = F.softmax(logits, dim=-1)
                score.extend(logits[:, 0].cpu().numpy().tolist())
                logits = logits.argmax(dim=-1)
                for t, p in zip(labels.view(-1), logits.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                logits = logits.cpu().numpy().tolist()
                res.extend(logits)
        p1 = confusion_matrix[1, 1] / confusion_matrix[:, 1].sum()
        r1 = confusion_matrix[1, 1] / confusion_matrix[1, :].sum()
        f1 = 2 * p1 * r1 / (p1 + r1)
        logger.info('预测数: {0:>5}, 正样本数: {1:>5}, 准确数:{2:>5}, f1: {3:>5.3}, p1: {4:>6.2%}, r1: {5:>6.2%}'.format(
            confusion_matrix[:, 1].sum(), confusion_matrix[1, :].sum(), confusion_matrix[1, 1], f1, p1, r1))
        self.model.train()
        return res, score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='../data/know/', type=str, help="The input data dir.")
    parser.add_argument("--pretrained_path", default='../pretrained_model/bert-base-chinese', type=str,
                        help='Path to pre-trained model')
    parser.add_argument("--save_path", default='trained3e5_warm.model', type=str, help="Path to save model.")

    parser.add_argument("--labels", default=[0, 1], type=list, help="labels")
    parser.add_argument("--device", default=0, type=int, help="device id")
    parser.add_argument("--require_improvement", default=10000000, type=int, help="early stopping steps")
    parser.add_argument("--num_epochs", default=4, type=int, help="epoch num")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--max_len", default=128, type=int, help="max length")
    parser.add_argument("--lr", default=3e-5, type=float, help="labels")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max gradient norm")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation step")
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='epsilon for Adam')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--cuda_devices', default=[], type=list, help='weight decay')
    parser.add_argument('--num_warmup_steps_ratio', default=0.1, type=float, help='ratio of warmup steps')
    parser.add_argument('--log_freq', default='epoch_end', type=str, help='frequency of log')
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--scheduler', type=str, default='CONSTANT', help="scheduler")
    parser.add_argument('--loss', type=str, default='ce', help="Loss function")

    args = parser.parse_args()
    args.num_classes = len(args.labels)
    args.num_batches = 0
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)

    if args.mode == 'train':
        train_data = load_dataset(args.data_dir + 'train2.txt', mode='train')
        dev_data = load_dataset(args.data_dir + 'dev_r.txt', mode='dev')
        test_data = load_dataset(args.data_dir + 'test.txt', mode='test')
        train_it = DataIterator(train_data, args.batch_size, rand=True)
        dev_it = DataIterator(dev_data, args.batch_size)
        test_it = DataIterator(test_data, args.batch_size)

        args.num_batches = len(train_it)
        set_seed(args.seed)
        cls_model = BertForSequenceClassification.from_pretrained(args.pretrained_path, num_labels=args.num_classes)
        trainer = KNOWTrainer(args, cls_model)
        trainer.train(train_it, dev_it, test_it)
        # 记录预测结果
        res, score = trainer.test(test_it)
        fw = open(args.data_dir + 'predict.txt', 'w')
        lines = open(args.data_dir + 'test.txt', 'r').readlines()
        logger.info('res num: {}, score num: {}, lines num: {}'.format(len(res), len(score), len(lines)))
        for p, i, line in zip(score, res, lines):
            line = line.strip('\n') + '\t' + str(i) + '\t' + str(p)
            fw.write(line + '\n')
        fw.close()
    elif args.mode == 'dev':
        p_list = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']
        train_data = []
        for i, path in enumerate(p_list):
            if i != args.val_id:
                train_data.extend(load_dataset(args.data_dir + path, mode='train'))
        dev_data = load_dataset(args.data_dir + p_list[args.val_id],  mode='dev')
        test_data = load_dataset(args.data_dir + 'test.txt', mode='test')
        test_it = DataIterator(test_data, args.batch_size)
        train_it = DataIterator(train_data, args.batch_size, rand=True)
        dev_it = DataIterator(dev_data, args.batch_size)
        args.num_batches = len(train_it)
        cls_model = BertForSequenceClassification.from_pretrained(args.pretrained_path, num_labels=args.num_classes)
        trainer = KNOWTrainer(args, cls_model)
        trainer.train(train_it, dev_it, test_it)
        res, score = trainer.test(dev_it, reload=True)
        fw = open(args.data_dir + 'predict' + str(args.val_id), 'w', encoding='utf8')
        lines = open(args.data_dir + p_list[args.val_id], 'r', encoding='utf8').readlines()
        logger.info('res num: {}, score num: {}, lines num: {}'.format(len(res), len(score), len(lines)))
        id2label = {i: lab for i, lab in enumerate(args.labels)}
        for p, i, line in zip(score, res, lines):
            line = line.strip('\n') + '\t' + str(i) + '\t' + str(p)
            fw.write(line + '\n')
        fw.close()

    else:
        set_seed(args.seed)
        cls_model = BertForSequenceClassification.from_pretrained(args.pretrained_path, num_labels=args.num_classes)
        trainer = KNOWTrainer(args, cls_model)
        test_data = load_dataset(args.data_dir + 'test.txt', mode='test')
        test_it = DataIterator(test_data, args.batch_size)
        res, score = trainer.test(test_it, reload=True)

        fw = open(args.data_dir + 'predict.txt', 'w')
        lines = open(args.data_dir + 'test.txt', 'r').readlines()
        logger.info('res num: {}, score num: {}, lines num: {}'.format(len(res), len(score), len(lines)))
        for p, i, line in zip(score, res, lines):
            line = line.strip('\n') + '\t' + str(i) + '\t' + str(p)
            fw.write(line + '\n')
        fw.close()


