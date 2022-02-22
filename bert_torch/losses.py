import sys
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """带 label smoothing正则化的交叉熵
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.size_average = size_average

    def forward(self, inputs, targets, use_label_smoothing=True):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data, 1)
        targets = targets.to(inputs.device)
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets.detach() * log_probs).sum(1)
        if self.size_average:
            return loss.mean()
        return loss


class FocalLoss(nn.Module):
    """来自 https://github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        # -at * (1-y_p)^gamma * log(y_p)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        # 不需要计算梯度
        # pt = Variable(logpt.data.exp())
        pt = logpt.detach().exp()
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            # 不需要计算梯度
            at = self.alpha.gather(0, target.data.view(-1))
            # logpt = logpt * Variable(at)
            logpt = logpt * at.detach()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SimCSELoss(nn.Module):
    """SimCSE loss 原文: https://github.com/princeton-nlp/SimCSE
    实现参考: https://zhuanlan.zhihu.com/p/378340148
    """
    def __init__(self, tem):
        super(SimCSELoss, self).__init__()
        self.tem = tem
        self.ce = nn.CrossEntropyLoss()

    def forward(self, batch_emb):
        batch_size = batch_emb.size(0)
        # 获取一个以相邻样本互为正样本的y_true 例如[1,0,3,2,5,4]
        y_true = torch.cat([torch.arange(1, batch_size, step=2, dtype=torch.long).unsqueeze(1),
                            torch.arange(0, batch_size, step=2, dtype=torch.long).unsqueeze(1)], dim=1)\
            .reshape([batch_size, ])
        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        sim_score = torch.matmul(norm_emb, norm_emb.T)
        sim_score = sim_score - torch.eye(batch_size) * 1e12  # 对角线为自身的相似度设为负无穷
        sim_score = sim_score * 20  # 除温度系数0.05
        loss = self.ce(sim_score, y_true)
        return loss


class SupConLossPLMS(torch.nn.Module):
    """Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning: https://arxiv.org/abs/2011.01403
    实现参考: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    """
    def __init__(self, device, temperature=0.05):
        super(SupConLossPLMS, self).__init__()
        self.tem = temperature
        self.device = device

    def forward(self, batch_emb, labels=None):
        labels = labels.view(-1, 1)
        batch_size = batch_emb.shape[0]
        mask = torch.eq(labels, labels.T).float()
        norm_emb = F.normalize(batch_emb, dim=1, p=2)
        # compute logits
        dot_contrast = torch.div(torch.matmul(norm_emb, norm_emb.T), self.tem)
        # for numerical stability
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)  # _返回索引
        logits = dot_contrast - logits_max.detach()
        # 索引应该保证设备相同
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_sum = mask.sum(1)
        # 防止出现NAN
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_sum
        return mean_log_prob_pos.mean()


