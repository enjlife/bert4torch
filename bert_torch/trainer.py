import torch
import torch.nn as nn
from torch.optim import Adam
from bert_torch import get_scheduler


class Trainer(object):

    def __init__(self, config, train_iter, valid_iter):
        cuda_condition = torch.cuda.is_available() and config.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.model = config.model.to(self.device)
        if config.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=config.cuda_devices)
        self.train_data = train_iter
        self.valid_data = valid_iter
        # --------------------------------------------------------------------
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = Adam(optimizer_grouped_parameters, lr=config.lr, betas=config.betas)
        self.scheduler = get_scheduler(config.scheduler, optimizer)
        # Using Negative Log Likelihood Loss function for predicting the masked_token
        # self.criterion = nn.NLLLoss(ignore_index=0)
        self.num_epochs = config.num_epochs
        self.log_freq = config.log_freq
        self.logger = config.logger
        self.logger.info("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self):
        raise NotImplementedError

    def valid(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save(self, epoch, save_path='trained.model', best=True):
        output_path = save_path if best else save_path + ".ep%d" % epoch
        torch.save(self.model.state_dict(), output_path)
        # self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path



