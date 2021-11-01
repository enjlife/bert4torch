# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging
import abc
import sys


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


import torch
import operator
import functools
from copy import copy
from math import sqrt


class AdaFactor(torch.optim.Optimizer):
    def __init__(self, params, lr=None, beta1=0.9, beta2=0.999, eps1=1e-30,
                 eps2=1e-3, cliping_threshold=1, non_constant_decay=True,
                 enable_factorization=True, ams_grad=True, weight_decay=0):

        enable_momentum = beta1 != 0
        self.beta1_glob = copy(beta1)
        self.beta2_glob = copy(beta2)
        self.lr_glob = copy(lr)

        beta1 = self.beta1_glob if hasattr(beta1, '__call__') else lambda x: self.beta1_glob
        beta2 = self.beta2_glob if hasattr(beta2, '__call__') else lambda x: self.beta2_glob

        if non_constant_decay:
            ams_grad = False
            if isinstance(self.beta1_glob, float):
                beta1 = lambda t: self.beta1_glob * (1 - self.beta1_glob ** (t - 1)) / (1 - self.beta1_glob ** t)
            if isinstance(self.beta2_glob, float):
                beta2 = lambda t: self.beta2_glob * (1 - self.beta2_glob ** (t - 1)) / (1 - self.beta2_glob ** t)

        relative_step_size = True

        if lr is None:
            # default value from article
            lr = lambda t: min(1e-2, 1 / sqrt(t))

        if isinstance(self.lr_glob, float):
            lr = lambda x: self.lr_glob
            relative_step_size = False

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps1=eps1,
                        eps2=eps2, cliping_threshold=cliping_threshold, weight_decay=weight_decay, ams_grad=ams_grad,
                        enable_factorization=enable_factorization,
                        enable_momentum=enable_momentum, relative_step_size=relative_step_size)

        super(AdaFactor, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaFactor, self).__setstate__(state)

    def _experimental_reshape(self, shape):
        temp_shape = shape[2:]
        if len(temp_shape) == 1:
            new_shape = (shape[0], shape[1] * shape[2])
        else:
            tmp_div = len(temp_shape) // 2 + len(temp_shape) % 2
            new_shape = (shape[0] * functools.reduce(operator.mul, temp_shape[tmp_div:], 1),
                         shape[1] * functools.reduce(operator.mul, temp_shape[:tmp_div], 1))
        return new_shape, copy(shape)

    def _check_shape(self, shape):
        '''
        output1 - True - algorithm for matrix, False - vector;
        output2 - need reshape
        '''
        if len(shape) > 2:
            return True, True
        elif len(shape) == 2:
            return True, False
        elif len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            return False, False
        else:
            return False, False

    def _rms(self, x):
        return sqrt(torch.mean(x.pow(2)))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                data_backup = p.data.clone().detach()

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                is_matrix, is_need_reshape = self._check_shape(grad.size())
                new_shape = p.data.size()
                if is_need_reshape and group['enable_factorization']:
                    new_shape, old_shape = \
                        self._experimental_reshape(p.data.size())
                    grad = grad.view(new_shape)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    if group['enable_momentum']:
                        state['exp_avg'] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)

                    if is_matrix and group['enable_factorization']:
                        state['exp_avg_sq_R'] = torch.zeros((1, new_shape[1]), dtype=torch.float32,
                                                            device=p.grad.device)
                        state['exp_avg_sq_C'] = torch.zeros((new_shape[0], 1), dtype=torch.float32,
                                                            device=p.grad.device)
                    else:
                        state['exp_avg_sq'] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)
                    if group['ams_grad']:
                        state['exp_avg_sq_hat'] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)

                if group['enable_momentum']:
                    exp_avg = state['exp_avg']

                if is_matrix and group['enable_factorization']:
                    exp_avg_sq_R = state['exp_avg_sq_R']
                    exp_avg_sq_C = state['exp_avg_sq_C']
                else:
                    exp_avg_sq = state['exp_avg_sq']

                if group['ams_grad']:
                    exp_avg_sq_hat = state['exp_avg_sq_hat']

                state['step'] += 1
                lr_t = group['lr'](state['step'])
                if group['relative_step_size']:
                    lr_t *= max(group['eps2'], self._rms(p.data))

                if group['enable_momentum']:
                    beta1_t = group['beta1'](state['step'])
                    exp_avg.mul_(beta1_t).add_(1 - beta1_t, grad)

                beta2_t = group['beta2'](state['step'])

                if is_matrix and group['enable_factorization']:
                    exp_avg_sq_R.mul_(beta2_t).add_(1 - beta2_t,
                                                    torch.sum(torch.mul(grad, grad).add_(group['eps1']), dim=0,
                                                              keepdim=True))
                    exp_avg_sq_C.mul_(beta2_t).add_(1 - beta2_t,
                                                    torch.sum(torch.mul(grad, grad).add_(group['eps1']), dim=1,
                                                              keepdim=True))
                    v = torch.mul(exp_avg_sq_C, exp_avg_sq_R).div_(torch.sum(exp_avg_sq_R))
                else:
                    exp_avg_sq.mul_(beta2_t).addcmul_(1 - beta2_t, grad, grad).add_((1 - beta2_t) * group['eps1'])
                    v = exp_avg_sq

                g = grad
                if group['enable_momentum']:
                    g = torch.div(exp_avg, 1 - beta1_t ** state['step'])

                if group['ams_grad']:
                    torch.max(exp_avg_sq_hat, v, out=exp_avg_sq_hat)
                    v = exp_avg_sq_hat
                    u = torch.div(g, (torch.div(v, 1 - beta2_t ** state['step'])).sqrt().add_(group['eps1']))
                else:
                    u = torch.div(g, v.sqrt())

                u.div_(max(1, self._rms(u) / group['cliping_threshold']))
                p.data.add_(-lr_t * (u.view(old_shape) if is_need_reshape and group['enable_factorization'] else u))

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * lr_t, data_backup)

        return loss


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class _LRSchedule(ABC):
    """ Parent of all LRSchedules here. """
    warn_t_total = False    # is set to True for schedules where progressing beyond t_total steps doesn't make sense

    def __init__(self, warmup=0.002, t_total=-1, **kw):
        """
        :param warmup:  what fraction of t_total steps will be used for linear warmup
        :param t_total: how many training steps (updates) are planned
        :param kw:
        """
        super(_LRSchedule, self).__init__(**kw)
        if t_total < 0:
            print("t_total value of {} results in schedule not being applied".format(t_total))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0] or -1".format(warmup))
        warmup = max(warmup, 0.)
        self.warmup, self.t_total = float(warmup), float(t_total)
        self.warned_for_t_total_at_progress = -1

    def get_lr(self, step, nowarn=False):
        """
        :param step:    which of t_total steps we're on
        :param nowarn:  set to True to suppress warning regarding training beyond specified 't_total' steps
        :return:        learning rate multiplier for current update
        """
        if self.t_total < 0:
            return 1.
        progress = float(step) / self.t_total
        ret = self.get_lr_(progress)
        # warning for exceeding t_total (only active with warmup_linear
        if not nowarn and self.warn_t_total and progress > 1. and progress > self.warned_for_t_total_at_progress:
            print(
                "Training beyond specified 't_total'. Learning rate multiplier set to {}. Please set 't_total' of {} correctly."
                    .format(ret, self.__class__.__name__))
            self.warned_for_t_total_at_progress = progress
        # end warning
        return ret

    @abc.abstractmethod
    def get_lr_(self, progress):
        """
        :param progress:    value between 0 and 1 (unless going beyond t_total steps) specifying training progress
        :return:            learning rate multiplier for current update
        """
        return 1.


class ConstantLR(_LRSchedule):
    def get_lr_(self, progress):
        return 1.


class WarmupCosineSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Decreases learning rate from 1. to 0. over remaining `1 - warmup` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    warn_t_total = True
    def __init__(self, warmup=0.002, t_total=-1, cycles=.5, **kw):
        """
        :param warmup:      see LRSchedule
        :param t_total:     see LRSchedule
        :param cycles:      number of cycles. Default: 0.5, corresponding to cosine decay from 1. at progress==warmup and 0 at progress==1.
        :param kw:
        """
        super(WarmupCosineSchedule, self).__init__(warmup=warmup, t_total=t_total, **kw)
        self.cycles = cycles

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)   # progress after warmup
            return 0.5 * (1. + math.cos(math.pi * self.cycles * 2 * progress))


class WarmupCosineWithHardRestartsSchedule(WarmupCosineSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
    learning rate (with hard restarts).
    """
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(warmup=warmup, t_total=t_total, cycles=cycles, **kw)
        assert(cycles >= 1.)

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)     # progress after warmup
            ret = 0.5 * (1. + math.cos(math.pi * ((self.cycles * progress) % 1)))
            return ret


class WarmupCosineWithWarmupRestartsSchedule(WarmupCosineWithHardRestartsSchedule):
    """
    All training progress is divided in `cycles` (default=1.) parts of equal length.
    Every part follows a schedule with the first `warmup` fraction of the training steps linearly increasing from 0. to 1.,
    followed by a learning rate decreasing from 1. to 0. following a cosine curve.
    """
    def __init__(self, warmup=0.002, t_total=-1, cycles=1., **kw):
        assert(warmup * cycles < 1.)
        warmup = warmup * cycles if warmup >= 0 else warmup
        super(WarmupCosineWithWarmupRestartsSchedule, self).__init__(warmup=warmup, t_total=t_total, cycles=cycles, **kw)

    def get_lr_(self, progress):
        progress = progress * self.cycles % 1.
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)     # progress after warmup
            ret = 0.5 * (1. + math.cos(math.pi * progress))
            return ret


class WarmupConstantSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Keeps learning rate equal to 1. after warmup.
    """
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1.


class WarmupLinearSchedule(_LRSchedule):
    """
    Linearly increases learning rate from 0 to 1 over `warmup` fraction of training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `1 - warmup` steps.
    """
    warn_t_total = True
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)


SCHEDULES = {
    None:       ConstantLR,
    "none":     ConstantLR,
    "warmup_cosine": WarmupCosineSchedule,
    "warmup_constant": WarmupConstantSchedule,
    "warmup_linear": WarmupLinearSchedule
}


# class BertAdam(Optimizer):
#     """Implements BERT version of Adam algorithm with weight decay fix.
#     Params:
#         lr: learning rate
#         warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
#         t_total: total number of training steps for the learning
#             rate schedule, -1  means constant learning rate of 1. (no warmup regardless of warmup setting). Default: -1
#         schedule: schedule to use for the warmup (see above).
#             Can be `'warmup_linear'`, `'warmup_constant'`, `'warmup_cosine'`, `'none'`, `None` or a `_LRSchedule` object (see below).
#             If `None` or `'none'`, learning rate is always kept constant.
#             Default : `'warmup_linear'`
#         b1: Adams b1. Default: 0.9
#         b2: Adams b2. Default: 0.999
#         e: Adams epsilon. Default: 1e-6
#         weight_decay: Weight decay. Default: 0.01
#         max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
#     """
#     def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
#                  b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0, **kwargs):
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
#         if not isinstance(schedule, _LRSchedule) and schedule not in SCHEDULES:
#             raise ValueError("Invalid schedule parameter: {}".format(schedule))
#         if not 0.0 <= b1 < 1.0:
#             raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
#         if not 0.0 <= b2 < 1.0:
#             raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
#         if not e >= 0.0:
#             raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
#         # initialize schedule object
#         if not isinstance(schedule, _LRSchedule):
#             schedule_type = SCHEDULES[schedule]
#             schedule = schedule_type(warmup=warmup, t_total=t_total)
#         else:
#             if warmup != -1 or t_total != -1:
#                 print("warmup and t_total on the optimizer are ineffective when _LRSchedule object is provided as schedule. "
#                                "Please specify custom warmup and t_total in _LRSchedule object.")
#         defaults = dict(lr=lr, schedule=schedule,
#                         b1=b1, b2=b2, e=e, weight_decay=weight_decay,
#                         max_grad_norm=max_grad_norm)
#         super(BertAdam, self).__init__(params, defaults)
#
#     def get_lr(self):
#         lr = []
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 if len(state) == 0:
#                     return [0]
#                 lr_scheduled = group['lr']
#                 lr_scheduled *= group['schedule'].get_lr(state['step'])
#                 lr.append(lr_scheduled)
#         return lr
#
#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#
#                 state = self.state[p]
#
#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values
#                     state['next_m'] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state['next_v'] = torch.zeros_like(p.data)
#
#                 next_m, next_v = state['next_m'], state['next_v']
#                 beta1, beta2 = group['b1'], group['b2']
#
#                 # Add grad clipping
#                 if group['max_grad_norm'] > 0:
#                     clip_grad_norm_(p, group['max_grad_norm'])
#
#                 # Decay the first and second moment running average coefficient
#                 # In-place operations to update the averages at the same time
#                 next_m.mul_(beta1).add_(1 - beta1, grad)
#                 next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
#                 update = next_m / (next_v.sqrt() + group['e'])
#
#                 # Just adding the square of the weights to the loss function is *not*
#                 # the correct way of using L2 regularization/weight decay with Adam,
#                 # since that will interact with the m and v parameters in strange ways.
#                 #
#                 # Instead we want to decay the weights in a manner that doesn't interact
#                 # with the m/v parameters. This is equivalent to adding the square
#                 # of the weights to the loss with plain (non-momentum) SGD.
#                 if group['weight_decay'] > 0.0:
#                     update += group['weight_decay'] * p.data
#
#                 lr_scheduled = group['lr']
#                 lr_scheduled *= group['schedule'].get_lr(state['step'])
#
#                 update_with_lr = lr_scheduled * update
#                 p.data.add_(-update_with_lr)
#
#                 state['step'] += 1
#
#                 # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
#                 # No bias correction
#                 # bias_correction1 = 1 - beta1 ** state['step']
#                 # bias_correction2 = 1 - beta2 ** state['step']
#
#         return loss