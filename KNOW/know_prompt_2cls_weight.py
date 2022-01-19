import argparse
import os
import random
import sys
import time
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append('../')
from bert_torch import DatasetBase, BertTokenizer, BertForTwoSequenceClassification, Trainer, time_diff, \
                        sequence_padding, set_seed, get_logger, BertForNSPSequenceClassification

logger = get_logger()
"""change log
1.将最后一层CLS转为last2 CLS 待验证
2.将验证集由dev_r.txt -> dev_rr.txt 
3.修改预测代码，不再输出label
4.添加 weight_loss
"""


def load_dataset(path, p=None, mode='train'):
    roberta = 'roberta' in args.pretrained_path  # 控制roberta的segment id
    logger.info('roberta: {}'.format(roberta))
    logger.info('Load data using pattern: {} from {}'.format(p, path))
    unused_list = ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]']
    PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'
    D = []
    label2id = {lab: i for i, lab in enumerate(args.labels)}
    label22id = {lab: i for i, lab in enumerate(args.labels2)}
    if p:
        # p_tokens = [CLS] + tokenizer.tokenize(p) + ['[unused1]', '[unused2]', '[unused3]', '[unused4]', SEP]
        p_tokens = [CLS] + tokenizer.tokenize(p) + [SEP]
        p_type_ids = [0] * len(p_tokens)
    else:  # p-tuning
        p_tokens = [CLS] + unused_list + [SEP]
        p_type_ids = [0] * len(p_tokens)
    fr = open(path, 'r', encoding='utf8')
    for line in tqdm(fr):
        line = line.strip('\n').split('\t')
        if (mode == 'train' and len(line) != 5) or (mode != 'train' and len(line) != 4):
            continue
        # mid,label,tag,content
        if mode == 'train':
            content, label, label2, weight = line[3], line[1], line[2], line[4]
        else:
            content, label, label2 = line[3], line[1], line[2]
        tokens = tokenizer.tokenize(content) + [SEP]
        if roberta:
            type_ids = p_type_ids + [0] * len(tokens)
        else:
            type_ids = p_type_ids + [1] * len(tokens)
        tokens = p_tokens + tokens
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if mode == 'train':
            D.append((token_ids, type_ids, label2id[label], label22id.get(label2, 0), 1.0 if label == '1' else 1.0 - float(weight)))
        else:
            D.append((token_ids, type_ids, label2id[label], label22id.get(label2, 0)))
    fr.close()
    if mode == 'train':
        random.shuffle(D)
    logger.info('Load {} data num: {}'.format(mode, len(D)))
    return D


class DataIterator(DatasetBase):

    def __init__(self, data_list, batch_size, mode='train', rand=False):
        super(DataIterator, self).__init__(data_list, batch_size, rand)
        self.mode = mode

    def _to_tensor(self, datas):
        token_ids = [data[0] for data in datas]
        type_ids = [data[1] for data in datas]
        token_ids = sequence_padding(token_ids)
        type_ids = sequence_padding(type_ids)
        token_ids = torch.LongTensor(token_ids)
        type_ids = torch.LongTensor(type_ids)
        labels = torch.LongTensor([data[2] for data in datas])
        labels2 = torch.tensor([data[3] for data in datas], dtype=torch.long)
        if self.mode == 'train':
            ws = torch.FloatTensor([data[4] for data in datas])
            return token_ids, type_ids, labels, labels2, ws

        return token_ids, type_ids, labels, labels2


class KNOWTrainer(Trainer):
    def __init__(self, config, model):
        self.labels = config.labels
        self.labels2 = config.labels2
        self.ratio = config.loss_ratio
        super(KNOWTrainer, self).__init__(config, model)

    def train(self, train_iter=None, dev_iter=None, test_iter=None):
        start_time = time.time()
        step = 0
        train_loss = 0.0
        # min_loss = float('inf')
        max_f1 = 0.0
        last_improve = 0
        flag = False  # 用于early-stop
        self.model.train()
        self.model.zero_grad()
        for epoch in range(self.num_epochs):
            logger.info('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))
            tq_iter = tqdm(train_iter, desc="EP_%s:%d" % ('train', epoch + 1), ncols=60)  # bar_format="{l_bar}{r_bar}"
            for (token_ids, type_ids, labels, labels2, ws) in tq_iter:
                token_ids, type_ids, labels, labels2, ws = token_ids.to(self.device), type_ids.to(self.device), \
                                                           labels.to(self.device), labels2.to(self.device), ws.to(self.device)
                logits, logits2 = self.model(token_ids, type_ids)
                ratio = 0.0
                # ratio = 0.2 - 0.1 * (step / self.t_total)  # min(0.8, max(0.2, (step+1)/self.t_total))
                loss = (F.cross_entropy(logits.view(-1, 2), labels.view(-1), reduction='none') * ws).mean() + \
                       ratio * self.loss_fct(logits2.view(-1, len(self.labels2)), labels2.view(-1))
                # loss = self.loss_fct(logits.view(-1, 2), labels.view(-1)) + ratio * self.loss_fct(logits2.view(-1, len(self.labels2)), labels2.view(-1))
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
                        improve = '***'
                        last_improve = step
                        save_path = self.save_model(step, best=False)
                        logger.info("EP:%d Model Saved on:%s" % (step, save_path))
                    else:
                        improve = ''
                    t_dif = time_diff(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>5.2}, Val Loss: {2:>5.2}, f1: {3:>5.3}, Val p0: {4:>6.2%}, Val r0: {5:>6.2%}, Time: {6} {7}'
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
            for (token_ids, type_ids, labels, _) in dev_iter:
                token_ids, type_ids, labels = token_ids.to(self.device), type_ids.to(self.device), labels.to(self.device)
                logits, logits2 = self.model(token_ids, type_ids)
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
            logger.info('Load model from {}'.format(model_path))
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        confusion_matrix = torch.zeros(2, 2)
        score = []
        res = []
        with torch.no_grad():
            for (token_ids, type_ids, labels, _) in test_iter:
                token_ids, type_ids, labels = token_ids.to(self.device), type_ids.to(self.device), labels.to(self.device)
                logits, logits2 = self.model(token_ids, type_ids)
                logits = F.softmax(logits, dim=-1)
                score.extend(logits[:, 0].cpu().numpy().tolist())
                logits = logits.argmax(dim=-1)
                for t, p in zip(labels.view(-1), logits.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                res.extend(logits.cpu().numpy().tolist())
        p0 = confusion_matrix[0, 0] / confusion_matrix[:, 0].sum()
        r0 = confusion_matrix[0, 0] / confusion_matrix[0, :].sum()
        f1 = 2 * p0 * r0 / (p0 + r0)
        logger.info('预测数: {0:>5}, 正样本数: {1:>5}, 准确数:{2:>5}, f1: {3:>5.3}, p0: {4:>6.2%}, r0: {5:>6.2%}'.format(
            confusion_matrix[:, 0].sum(), confusion_matrix[0, :].sum(), confusion_matrix[0, 0], f1, p0, r0))
        self.model.train()
        return res, score


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
    parser.add_argument('--loss_ratio', type=float, default=0.8, help="ratio of loss")
    parser.add_argument('--nsp', type=bool, default=False, help="if nsp")
    parser.add_argument('--weight_loss', type=bool, default=False, help="weight_loss")

    args = parser.parse_args()
    args.num_classes = len(args.labels)
    args.num_batches = 0
    args.labels2 = '养生,法律,财经,植物,旅游出行,美女,孕产,汽车,搞笑,数码,音乐,it产业,医疗,人文艺术,宠物,体育,民俗,it技术,科学科普,游戏,' \
                   '三农,摄影拍照,休闲娱乐,电视剧,家装家居,综艺,婚庆,国学,教育,社会,军事,舞蹈,动物,娱乐明星,时政,政府政务,帅哥,读书作家,' \
                   '美妆,交通服务,星座命理,美食,育儿,历史,宗教,时尚,日常生活,小技巧,动漫,运动健身,情感,收藏,电影,房地产,职场,奢侈品,设计'.split(',')
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path, do_lower_case=True)  # dir is OK

    patterns = ['学到很多！', '知识，技巧。', '学会啦！', '小知识。', '这是一篇知识或技巧的博客。',  None]

    if args.mode == 'train':
        set_seed(args.seed)
        train_data = load_dataset(args.data_dir + 'train_logit.txt', p=patterns[args.pattern])
        dev_data = load_dataset(args.data_dir + 'dev_rr.txt', p=patterns[args.pattern], mode='dev')
        test_data = load_dataset(args.data_dir + 'test.txt', p=patterns[args.pattern], mode='test')
        train_it = DataIterator(train_data, args.batch_size, rand=True)
        dev_it = DataIterator(dev_data, args.batch_size, mode='dev')
        test_it = DataIterator(test_data, args.batch_size, mode='test')
        args.num_batches = len(train_it)
        if args.nsp:
            logger.info('Use NSP')
            cls_model = BertForNSPSequenceClassification.from_pretrained(args.pretrained_path, len(args.labels2))
        else:
            logger.info('Use 2CLS')
            cls_model = BertForTwoSequenceClassification.from_pretrained(args.pretrained_path, args.num_classes, len(args.labels2))
        trainer = KNOWTrainer(args, cls_model)
        trainer.train(train_it, dev_it, test_it)
        res, score = trainer.test(test_it, reload=True)
        fw = open(args.data_dir + 'predict2.txt', 'w', encoding='utf8')
        lines = open(args.data_dir + 'test.txt', 'r', encoding='utf8').readlines()
        logger.info('res num: {}, score num: {}, lines num: {}'.format(len(res), len(score), len(lines)))
        id2label = {i: lab for i, lab in enumerate(args.labels)}
        for p, i, line in zip(score, res, lines):
            line = line.strip('\n') + '\t' + id2label[i] + '\t' + str(p)
            fw.write(line + '\n')
        fw.close()
    # 交叉验证
    elif args.mode == 'dev':
        p_list = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']
        train_data = []
        for i, path in enumerate(p_list):
            if i != args.val_id:
                train_data.extend(load_dataset(args.data_dir + path, p=patterns[args.pattern], mode='train'))
        dev_data = load_dataset(args.data_dir + p_list[args.val_id], p=patterns[args.pattern], mode='dev')
        test_data = load_dataset(args.data_dir + 'test.txt', p=patterns[args.pattern], mode='test')
        test_it = DataIterator(test_data, args.batch_size)
        train_it = DataIterator(train_data, args.batch_size, rand=True)
        args.num_batches = len(train_it)
        dev_it = DataIterator(dev_data, args.batch_size)
        cls_model = BertForNextSentencePrediction.from_pretrained(args.pretrained_path)
        trainer = KNOWTrainer(args, cls_model)
        trainer.train(train_it, dev_it, test_it)
        res, score = trainer.test(dev_it, reload=True)
        fw = open(args.data_dir + 'predict' + str(args.val_id), 'w', encoding='utf8')
        lines = open(args.data_dir + p_list[args.val_id], 'r', encoding='utf8').readlines()
        logger.info('res num: {}, score num: {}, lines num: {}'.format(len(res), len(score), len(lines)))
        id2label = {i: lab for i, lab in enumerate(args.labels)}
        for p, i, line in zip(score, res, lines):
            line = line.strip('\n') + '\t' + id2label[i] + '\t' + str(p)
            fw.write(line + '\n')
        fw.close()

    elif args.mode == 'test':
        set_seed(args.seed)
        weights = torch.load(args.save_path)
        cls_model = BertForTwoSequenceClassification.from_pretrained(args.pretrained_path, args.num_classes, len(args.labels2), state_dict=weights)
        trainer = KNOWTrainer(args, cls_model)
        test_data = load_dataset(args.data_dir + 'test.txt', p=patterns[args.pattern], mode='test')
        test_it = DataIterator(test_data, args.batch_size, mode='test')
        res, score = trainer.test(test_it, reload=False)
        fw = open(args.data_dir + 'predict_test.txt', 'w', encoding='utf8')
        lines = open(args.data_dir + 'test.txt', 'r', encoding='utf8').readlines()
        logger.info('res num: {}, score num: {}, lines num: {}'.format(len(res), len(score), len(lines)))
        id2label = {i: lab for i, lab in enumerate(args.labels)}
        for p, i, line in zip(score, res, lines):
            line = line.strip('\n') + '\t' + str(p)
            fw.write(line + '\n')
        fw.close()

    elif args.mode == 'predict':
        device = torch.device(args.device)
        logger.info('device: {}'.format(device))
        cls_model = BertForNextSentencePrediction.from_pretrained(args.pretrained_path)
        cls_model.load_state_dict(torch.load(args.save_path))
        cls_model.to(device)
        cls_model.eval()
        files = os.listdir(args.data_dir)
        files = sorted(files, reverse=False)
        for file in files:
            start_time = time.time()
            predict_data, mids = load_dataset_predict(args.data_dir + file, p=patterns[args.pattern], mode='test')
            test_it = DataIterator(predict_data, args.batch_size)
            t_dif = time_diff(start_time)
            logger.info('Load data time: {}'.format(t_dif))
            start_time = time.time()
            scores = []
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




