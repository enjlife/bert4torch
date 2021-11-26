import torch
import torch.nn as nn
from torch.optim import Adam
from .optimizers import get_scheduler
from reference.local_logging import get_logger, _get_library_root_logger

logger = _get_library_root_logger()


class Trainer(object):

    def __init__(self, config, model):
        cuda_condition = torch.cuda.is_available() and config.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.model = model.to(self.device)
        if config.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=config.cuda_devices)
        self.train_data = config.train_iter if hasattr(config, 'train_iter') else []
        self.dev_data = config.dev_iter if hasattr(config, 'dev_iter') else []

        self.num_epochs = config.num_epochs
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=config.lr, betas=config.betas)
        num_warmup_steps = self.num_epochs * len(self.train_data) * config.num_warmup_steps_ratio
        self.scheduler = get_scheduler(config.scheduler, self.optimizer, num_warmup_steps)
        self.save_path = config.save_path if hasattr(config, 'save_path') else 'trained.model'
        self.log_freq = config.log_freq
        logger.info("Total Parameters: %d" % sum([p.nelement() for p in self.model.parameters()]))

    def train(self):
        raise NotImplementedError

    def dev(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save_model(self, epoch, best=True):
        output_path = self.save_path if best else self.save_path + ".ep%d" % epoch
        torch.save(self.model.state_dict(), output_path)
        # self.model.to(self.device)
        logger.info("EP:%d Model Saved on:%s" % (epoch, output_path))
        return output_path



