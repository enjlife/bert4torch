import sys
import fasttext
import numpy as np
import argparse


def test(model, path, save_path):
    lines = open(path, 'r').readlines()
    lines = [line.rstrip('\n').split('\t') for line in lines]

    mids = [line[0] for line in lines]
    texts = [line[1].split(' ') for line in lines]
    labels = [x[0] for x in texts]
    datas = [' '.join(x[1:]) for x in texts]
    p_labels, logits = model.predict(datas, k=2)

    print('datas: {}, labels: {}, logits: {}'.format(len(texts), len(labels), len(logits)), flush=True)
    confusion_matrixs = [np.zeros((2, 2)) for _ in range(len(args.thresholds))]  # 同一个位置

    for label, logit in zip(labels, logits):
        for i in range(len(confusion_matrixs)):
            p = 1 if logit[1] >= args.thresholds[i] else 0
            confusion_matrixs[i][int(label)][p] += 1

    for i in range(len(confusion_matrixs)):
        p1 = confusion_matrixs[i][1, 1] / confusion_matrixs[i][:, 1].sum()
        r1 = confusion_matrixs[i][1, 1] / confusion_matrixs[i][1, :].sum()
        f1 = 2 * p1 * r1 / (p1 + r1)
        print('预测数: {0:>5}, 正样本数: {1:>5}, 准确数:{2:>5}, f1: {3:>5.3}, p1: {4:>6.2%}, r1: {5:>6.2%}'.format(
            confusion_matrixs[i][:, 1].sum(), confusion_matrixs[i][1, :].sum(), confusion_matrixs[i][1, 1], f1, p1, r1),flush=True)

        if p1 > args.acc:
            args.acc = p1
            fw = open(save_path, 'w')
            for mid, label, text, logit in zip(mids, labels, texts, logits):
                text = ''.join(text[1:])
                fw.write('{}\t{}\t{}\t{:>5.3}\n'.format(mid, label, text, logit[1]))
            fw.close()
            model.save_model(args.model_save_path)


def train():
    # train_supervised uses the same arguments and defaults as the fastText cli
    # loss='hs' bucket=2000000
    print('lr: {}, num_epoch: {}'.format(args.lr, args.num_epoch), flush=True)
    model = fasttext.train_supervised(args.train_path, label_prefix='__label__', epoch=args.num_epoch, lr=args.lr,
                                      wordNgrams=args.wordngrams, verbose=2, minCount=args.mincount)
    test(model, args.dev_path, save_path=args.dev_path+'.res')
    test(model, args.test_path, save_path=args.test_path+'.res')
    model.save_model(args.model_save_path)


def search():
    for lr in args.lrs:
        for num_epoch in args.num_epochs:
            print('lr: {}, num_epoch: {}'.format(lr, num_epoch), flush=True)
            model = fasttext.train_supervised(args.train_path, label_prefix='__label__', epoch=num_epoch,
                                              lr=lr, wordNgrams=args.wordngrams, verbose=2, minCount=args.mincount)
            test(model, args.dev_path, save_path=args.dev_path+'.res')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Train or test')
    parser.add_argument("--num_epoch", default=25, type=int, help="Epoch num")
    parser.add_argument("--lr", default=1.0, type=float, help="learning rate")
    parser.add_argument('--train_path', default='train_ft.txt', type=str, help='Train path')
    parser.add_argument('--dev_path', default='dev_ft.txt', type=str, help='Dev path')
    parser.add_argument('--test_path', default='test_ft.txt', type=str, help='Test path')
    parser.add_argument('--thresholds', default='0.25', type=str, help='阈值列表')
    parser.add_argument('--model_save_path', default='ft_model.bin', type=str, help='Model save path')
    parser.add_argument('--wordngrams', default=5, type=int, help='Ngram')
    parser.add_argument('--mincount', default=8, type=int, help='Min count')

    args = parser.parse_args()
    if args.mode == 'train':
        args.thresholds = list(map(float, args.thresholds.split(',')))
        print(args.thresholds)
        train()
    elif args.mode == 'dev':
        model = fasttext.load_model(args.model_save_path)
        test(model, args.test_path, args.test_path+'res')
    elif args.mode == 'search':
        args.acc = 0.0
        args.thresholds = [0.25]  # [0.05, 0.1, 0.15, 0.2, 0.25]
        args.lrs = [0.1 * i for i in range(1, 11)]
        args.num_epochs = list(range(5, 40))
        # args.thresholds = [0.25]
        # args.lrs = [0.6, 0.7, 1.0]
        # args.num_epochs = [8, 9, 15]
        search()


