import argparse
import os
import random
import sys
import time
import json
import re
import six
import pandas as pd
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
from multiprocessing import Queue, Process
sys.path.append('../')
from bert_torch import DatasetBase, BertTokenizer, BertForNextSentencePrediction, Trainer, time_diff, \
                        sequence_padding, set_seed, get_logger

logger = get_logger()


def load_dataset(path, p=None, mode='train'):
    logger.info('Load data using pattern: {} from {}'.format(p, path))
    unused_list = ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]']
    PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'
    D = []
    label2id = {lab: i for i, lab in enumerate(args.labels)}
    if p:
        p_tokens = [CLS] + tokenizer.tokenize(p) + [SEP]
        p_type_ids = [0] * len(p_tokens)
    else:  # p-tuning
        p_tokens = [CLS] + unused_list + [SEP]
        p_type_ids = [0] * len(p_tokens)
    fr = open(path, 'r', encoding='utf8')
    for line in tqdm(fr):
        line = line.strip('\n').split('\t')
        if len(line) != 4:
            continue
        # mid,label,tag,content
        content, label = line[-1], line[1]
        tokens = tokenizer.tokenize(content) + [SEP]
        type_ids = p_type_ids + [1] * len(tokens)
        tokens = p_tokens + tokens
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        D.append((token_ids, type_ids, label2id[label]))
    fr.close()
    if mode == 'train':
        random.shuffle(D)
    logger.info('Load {} data num: {}'.format(mode, len(D)))
    return D


def preprocess(d, max_len=160):
    text, l_title, v_title, v_info = d['TEXT'], d.get('LURL_TITLE', ''), d.get('VIDEO_TITLE', ''), \
                                          eval(d.get('VIDEO_INFO', '{}')).get('display_name', '')
    if l_title and l_title not in text:
        text += l_title
    if v_title and v_title not in text:
        text += v_title
    if v_info and v_info not in text:
        text += v_info
    if six.PY2:
        text = text.lower().decode()
    else:
        text = text.lower()
    text = re.sub(u'(https{0,1}://[a-zA-Z0-9/\.\-]*)|(\<.*?\>)|(@\S+)|(\[.*?\])|(我的评分：)', '', text)
    text = text.strip()
    text = re.sub(u'[\t\n\r ]', '，', text)
    text = ''.join(re.findall(u'[一-龥\w（）〈〉《》「」『』﹃﹄〔〕…—～﹏￥、【】，。？！：；“”‘]', text))
    text = re.sub(u'([？。！，])([？，。！]{1,})', '\\1', text)

    return text[:max_len]


def load_dataset_predict(path, p=None, mode='train', max_len=160):
    logger.info('Load data using pattern: {} from {}'.format(p, path))
    unused_list = ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]']
    PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'
    D, MIDS = [], []
    if p:
        p_tokens = [CLS] + tokenizer.tokenize(p) + [SEP]
        p_type_ids = [0] * len(p_tokens)
    else:  # p-tuning
        p_tokens = [CLS] + unused_list + [SEP]
        p_type_ids = [0] * len(p_tokens)
    fr = open(path, 'r', encoding='utf8')
    i = 0
    for line in tqdm(fr):
        line = line.strip()
        if not line:
            continue
        d = eval(line)
        if not d or 'TEXT' not in d:
            continue
        text = preprocess(d, max_len=max_len)
        mid = d['ID']
        if i < 100:
            print('{}\t{}'.format(mid, text))
        if len(text) < 5:
            continue
        tokens = tokenizer.tokenize(text) + [SEP]
        type_ids = p_type_ids + [1] * len(tokens)
        tokens = p_tokens + tokens
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        D.append((token_ids, type_ids, 0))
        MIDS.append(str(mid))
        i += 1
    fr.close()
    logger.info('Data num: {}, mid num: {}'.format(len(D), len(MIDS)))
    return D, MIDS


def write(q, file, p, batch_size):
    print('Process to write: %s' % os.getpid())
    predict_data, mids = load_dataset_predict(file, p=p)
    test_it = DataIterator(predict_data, batch_size)
    q.put((test_it, mids))


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
        min_loss = float('inf')
        max_f1 = 0.0
        last_improve = 0
        flag = False  # 用于early-stop
        self.model.train()
        self.model.zero_grad()
        for epoch in range(self.num_epochs):
            logger.info('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))
            tq_iter = tqdm(train_iter, desc="EP_%s:%d" % ('train', epoch + 1))  # bar_format="{l_bar}{r_bar}"
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
                    dev_loss, p0, r0, f1 = self.dev(dev_iter=dev_iter)
                    # if dev_loss < min_loss:
                    if f1 > max_f1:
                        max_f1 = f1
                        self.save_model(epoch)
                        improve = '^^^'
                        last_improve = step
                    else:
                        improve = ''
                    t_dif = time_diff(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Val Loss: {2:>5.2}, f1: {3:>5.2}, Val p0: {4:>6.2%}, Val r0: {5:>6.2%}, Time: {6} {7}'
                    logger.info(msg.format(step, train_loss, dev_loss, f1, p0, r0, t_dif, improve))
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
        # 计算验证集上的损失和准确率
        self.model.eval()
        confusion_matrix = torch.zeros(2, 2)
        loss_total = 0.0
        with torch.no_grad():
            for (token_ids, type_ids, labels) in dev_iter:
                token_ids, type_ids, labels = token_ids.to(self.device), type_ids.to(self.device), labels.to(self.device)
                logits = self.model(token_ids, type_ids)
                loss = self.loss_fct(logits.view(-1, 2), labels.view(-1))
                loss_total += loss.item()
                logits = logits.argmax(dim=-1)
                for t, p in zip(labels.view(-1), logits.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        self.model.train()
        p0 = confusion_matrix[0, 0] / confusion_matrix[:, 0].sum()
        r0 = confusion_matrix[0, 0] / confusion_matrix[0, :].sum()
        f1 = 2 * p0 * r0 / (p0 + r0)
        return loss_total/len(dev_iter), p0, r0, f1

    def test(self, test_iter, model_path=None, reload=False):
        if not model_path:
            model_path = self.save_path
        if reload:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        confusion_matrix = torch.zeros(2, 2)
        score = []
        res = []
        with torch.no_grad():
            for (token_ids, type_ids, labels) in test_iter:
                token_ids, type_ids, labels = token_ids.to(self.device), type_ids.to(self.device), labels.to(self.device)
                logits = self.model(token_ids, type_ids)
                logits = F.softmax(logits, dim=-1)
                score.extend(logits[:, 0].cpu().numpy().tolist())
                logits = logits.argmax(dim=-1)
                for t, p in zip(labels.view(-1), logits.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                res.extend(logits.cpu().numpy().tolist())
        p0 = confusion_matrix[0, 0] / confusion_matrix[:, 0].sum()
        r0 = confusion_matrix[0, 0] / confusion_matrix[0, :].sum()
        f1 = 2 * p0 * r0 / (p0 + r0)
        logger.info('预测数: {0:>5}, 正样本数: {1:>5}, 准确数:{2:>5}, f1: {3:>5.2}, p0: {4:>6.2%}, r0: {5:>6.2%}'.format(
            confusion_matrix[:, 0].sum(), confusion_matrix[0, :].sum(), confusion_matrix[0, 0], f1, p0, r0))
        self.model.train()
        return res, score


def cal_acc(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df[4] = df.apply(lambda x: x[2] == 1 and x[3] == 1, axis=1)
    return df[2].sum(), df[3].sum(), df[4].sum(), df[4].sum() / df[3].sum(), df[4].sum() / df[2].sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='../data/know/', type=str, help="The input data dir.")
    parser.add_argument("--pretrained_path", default='../pretrained_model/bert-base-chinese', type=str, help='Path to pre-trained model')
    parser.add_argument("--save_path", default='trained_prompt.model', type=str, help="Path to save model.")

    parser.add_argument("--labels", default=['1', '0'], type=list, help="Labels")
    parser.add_argument("--device", default=0, type=int, help="Device id")
    parser.add_argument("--require_improvement", default=10000000, type=int, help="Early stopping steps")
    parser.add_argument("--num_epochs", default=4, type=int, help="Epoch num")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--max_len", default=128, type=int, help="Max length")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation step")
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='Epsilon for Adam')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    parser.add_argument('--cuda_devices', default=[], type=list, help='List of multi gpu')
    parser.add_argument('--num_warmup_steps_ratio', default=0.1, type=float, help='Ratio of warmup steps')
    parser.add_argument('--log_freq', default='epoch_end', type=str, help='Frequency of log')
    parser.add_argument('--mode', default='train', type=str, help='Train or test')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for initialization")
    parser.add_argument('--scheduler', type=str, default='CONSTANT', help="Scheduler")
    parser.add_argument('--pattern', type=int, default=0, help="Pattern index")
    parser.add_argument('--loss', type=str, default='ce', help="Loss function")
    parser.add_argument('--val_id', type=int, default=0, help="validation id")

    args = parser.parse_args()
    args.num_classes = len(args.labels)
    args.num_batches = 0
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)  # dir is OK

    patterns = ['学到很多！', '知识，技巧。', '学会啦！', '小知识。', '这是一篇知识或技巧的博客。',  None]

    if args.mode == 'predict':
        device = torch.device(args.device)
        logger.info('device: {}'.format(device))
        cls_model = BertForNextSentencePrediction.from_pretrained(args.pretrained_path)
        cls_model.load_state_dict(torch.load(args.save_path))
        cls_model.to(device)
        cls_model.eval()
        files = os.listdir(args.data_dir)
        files = sorted(files, reverse=False)
        q = Queue()
        write(q, file=args.data_dir+files[0], p=patterns[args.pattern], batch_size=args.batch_size)
        for i, file in enumerate(files):
            start_time = time.time()
            if i < len(files) - 1:
                pw = Process(target=write, args=(q, args.data_dir_file, patterns[args.pattern], args.batch_size))
                pw.start()
            t_dif = time_diff(start_time)
            logger.info('Init data process: {}'.format(t_dif))
            start_time = time.time()
            scores = []
            test_it, mids = q.get()
            with torch.no_grad():
                for (token_ids, type_ids, _) in tqdm(test_it):
                    token_ids, type_ids = token_ids.to(device), type_ids.to(device)
                    logits = cls_model(token_ids, type_ids)
                    logits = F.softmax(logits, dim=-1)
                    scores.extend(logits[:, 0].cpu().numpy().tolist())
            logger.info('scores: {}, mids: {}'.format(len(scores), len(mids)))
            fw = open(args.data_dir + 'predict_' + file, 'w', encoding='utf8')
            for mid, score in zip(mids, scores):
                fw.write('{}\t{}\n'.format(mid, str(round(score, 3))))
            t_dif = time_diff(start_time)
            logger.info('Execute time: {}'.format(t_dif))
            if i < len(files) - 1:
                pw.join()




