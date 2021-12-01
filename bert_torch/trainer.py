import torch
import logging
import torch.nn as nn
from torch.optim import Adam
from .optimizers import get_scheduler, AdamW

logger = logging.getLogger()


class Trainer(object):

    def __init__(self, config, model):
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and config.with_cuda) else "cpu")
        self.model = model.to(self.device)
        if config.with_cuda and torch.cuda.device_count() > 1:
            logger.info("Using %d GPUS for model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=config.cuda_devices)
        self.train_data = config.train_iter if hasattr(config, 'train_iter') else []
        self.dev_data = config.dev_iter if hasattr(config, 'dev_iter') else []
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, betas=config.betas)

        self.num_epochs = config.num_epochs
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        t_total = self.num_epochs * len(self.train_data) // self.gradient_accumulation_steps
        # 计算warmup的步数 num_warmup_steps = t_total * num_warmup_steps_ratio (默认0.1)
        self.scheduler = get_scheduler(config.scheduler, self.optimizer, t_total * config.num_warmup_steps_ratio, t_total)
        self.max_grad_norm = config.max_grad_norm
        self.save_path = config.save_path if hasattr(config, 'save_path') else 'trained.model'
        self.log_freq = config.log_freq
        logger.info("Total Parameters: %d" % sum([p.nelement() for p in self.model.parameters()]))

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def dev(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, epoch, best=True):
        output_path = self.save_path if best else self.save_path + ".ep%d" % epoch
        torch.save(self.model.state_dict(), output_path)
        # self.model.to(self.device)
        logger.info("EP:%d Model Saved on:%s" % (epoch, output_path))
        return output_path



