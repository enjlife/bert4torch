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
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# 学习率不变
def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_decay_schedule(optimizer: Optimizer, num_warmup_steps: int, num_training_steps,  last_epoch: int = -1):
    """freeze_step = num_warmup_step"""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return 1.0
        return (num_training_steps - current_step) / max(1.0, (num_training_steps - num_warmup_steps))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# 线性增加，然后不变
def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# 学习率线性增加，线性降低
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 线性增加，余弦变化
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """num_cycles控制形状，默认值刚好从1~0
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 余弦退火 cosine with hard restarts
def get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=1,
                                                       last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# polynomial
def get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0,
                                              last_epoch=-1):

    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


SCHEDULER_FUNCTION = {
    'LINEAR': get_linear_schedule_with_warmup,
    'COSINE': get_cosine_schedule_with_warmup,
    'COSINE_WITH_RESTARTS': get_cosine_with_hard_restarts_schedule_with_warmup,
    'POLYNOMIAL': get_polynomial_decay_schedule_with_warmup,
    'CONSTANT': get_constant_schedule,  # 学习率不变
    'CONSTANT_WITH_WARMUP': get_constant_schedule_with_warmup,  # 线性增加，然后不变
    'CONSTANT_WITH_DECAY': get_constant_decay_schedule,  # 不变，然后线性降低
}


def get_scheduler(name, optimizer, num_warmup_steps=None, num_training_steps=None):
    if name not in SCHEDULER_FUNCTION:
        raise ValueError(f"{name} not in SCHEDULER_FUNCTION.")
    schedule_func = SCHEDULER_FUNCTION[name]
    if name == 'CONSTANT':
        return schedule_func(optimizer)
    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")
    if name == 'CONSTANT_WITH_WARMUP':
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


class AdamW(Optimizer):
    """带权重衰减的Adam
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)  # 将parmas添加到 self.param_groups

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure: 计算loss和梯度，返回loss。例如:
            ```
            for input, target in dataset:
                def closure():
                    optimizer.zero_grad()
                    output = model(input)
                    loss = loss_fn(output, target)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            ```
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

                # 初始化state
                if len(state) == 0:
                    state['step'] = 0
                    # 滑动平均--动量 Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 滑动平均--二阶原点矩 Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                # 原地操作
                # m_t = beta_1 * m_{t-1} + (1-beta_1) * g_t
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                # v_t = beta_2 * v_{t-1} + (1-beta_2) * g_t^2 + eps
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    # 学习率 * sqrt(1.0 - beta2^step) / (1.0 - beta1^step)
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                # theta_t - lr * m_t / v_t
                p.data.addcdiv_(-step_size, exp_avg, denom)
                # 将权重的平方加到loss对于Adam优化器并不合理，因为L2正则会与动量m和二阶矩v 产生交互
                # 在更新梯度最后添加权重衰减，与没有动量的SGD等效
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss

# class AdamW(Optimizer):
#     """
#     Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
#     <https://arxiv.org/abs/1711.05101>`__.
#
#     Parameters:
#         params (:obj:`Iterable[nn.parameter.Parameter]`):
#             Iterable of parameters to optimize or dictionaries defining parameter groups.
#         lr (:obj:`float`, `optional`, defaults to 1e-3):
#             The learning rate to use.
#         betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
#             Adam's betas parameters (b1, b2).
#         eps (:obj:`float`, `optional`, defaults to 1e-6):
#             Adam's epsilon for numerical stability.
#         weight_decay (:obj:`float`, `optional`, defaults to 0):
#             Decoupled weight decay to apply.
#         correct_bias (:obj:`bool`, `optional`, defaults to `True`):
#             Whether or not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
#     """
#
#     def __init__(
#         self,
#         params: Iterable[nn.parameter.Parameter],
#         lr: float = 1e-3,
#         betas: Tuple[float, float] = (0.9, 0.999),
#         eps: float = 1e-6,
#         weight_decay: float = 0.0,
#         correct_bias: bool = True,
#     ):
#         # require_version("torch>=1.5.0")  # add_ with alpha
#         if lr < 0.0:
#             raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
#         if not 0.0 <= eps:
#             raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
#         super().__init__(params, defaults)
#
#     def step(self, closure: Callable = None):
#         """
#         Performs a single optimization step.
#
#         Arguments:
#             closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
#
#                 state = self.state[p]
#
#                 # State initialization
#                 if len(state) == 0:
#                     state["step"] = 0
#                     # Exponential moving average of gradient values
#                     state["exp_avg"] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state["exp_avg_sq"] = torch.zeros_like(p.data)
#
#                 exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
#                 beta1, beta2 = group["betas"]
#
#                 state["step"] += 1
#
#                 # Decay the first and second moment running average coefficient
#                 # In-place operations to update the averages at the same time
#                 exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
#                 denom = exp_avg_sq.sqrt().add_(group["eps"])
#
#                 step_size = group["lr"]
#                 if group["correct_bias"]:  # No bias correction for Bert
#                     bias_correction1 = 1.0 - beta1 ** state["step"]
#                     bias_correction2 = 1.0 - beta2 ** state["step"]
#                     step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
#
#                 p.data.addcdiv_(exp_avg, denom, value=-step_size)
#
#                 # Just adding the square of the weights to the loss function is *not*
#                 # the correct way of using L2 regularization/weight decay with Adam,
#                 # since that will interact with the m and v parameters in strange ways.
#                 #
#                 # Instead we want to decay the weights in a manner that doesn't interact
#                 # with the m/v parameters. This is equivalent to adding the square
#                 # of the weights to the loss with plain (non-momentum) SGD.
#                 # Add weight decay at the end (fixed version)
#                 if group["weight_decay"] > 0.0:
#                     p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))
#
#         return loss


class Adafactor(Optimizer):
    """实现来自 transformers.optimization.py.Adafactor
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,  # RMS截断的参数
        decay_rate=-0.8,  # 计算beta
        beta1=None,  # 一阶滑动平均的系数,用于确定是否采用一阶动量
        weight_decay=0.0,  # 权重衰减
        scale_parameter=True,  # 梯度标准化
        relative_step=False,  # 自行计算学习率
        warmup_init=False,  # 使用自行计算学习率，学习率初始化方式
    ):
        # require_version("torch>=1.5.0")  # add_ with alpha
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        # 使用relative_step lr= min(1e-2 or 1e-6 * step, 1/sqrt(step))
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        # 梯度标准化 max(epsilon2, RMS(theta))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        """1.判断梯度矩阵形状 2.判断是否使用一阶滑动平均"""
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        """模长的变种: \sqrt(1\n \sum_{i=1}^n x_i^2)
        """
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        """计算梯度的近似矩阵：行向量和列向量为均值，相当于近似矩阵除mn,原矩阵同除mn即可
        """
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_()
        c_factor = exp_avg_sq_col.rsqrt()
        return torch.mm(r_factor.unsqueeze(-1), c_factor.unsqueeze(0))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape
                # 二阶矩是否分解 是否计算一阶动量
                factored, use_first_moment = self._get_options(group, grad_shape)
                # 初始化state
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        # 一维梯度不分解
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                # RMS(theta)用于梯度标准化
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)
                # \beta_{2,t} = 1 - t^{c}
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]  # epsilon1
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
                    # 计算二阶原点矩的近似矩阵 update=1/sqrt(v_{t})
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss


class AdafactorSchedule(LambdaLR):
    """
    Since :class:`~transformers.optimization.Adafactor` performs its own scheduling, if the training loop relies on a
    scheduler (e.g., for logging), this class creates a proxy object that retrieves the current lr values from the
    optimizer.

    It returns ``initial_lr`` during startup and the actual ``lr`` during stepping.
    """

    def __init__(self, optimizer, initial_lr=0.0):
        def lr_lambda(_):
            return initial_lr

        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr
        super().__init__(optimizer, lr_lambda)
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        opt = self.optimizer
        lrs = [
            opt._get_lr(group, opt.state[group["params"][0]])
            for group in opt.param_groups
            if group["params"][0].grad is not None
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs

# class ExplicitEnum(Enum):
#     """
#     Enum with more explicit error message for missing values.
#     """
#     @classmethod
#     def _missing_(cls, value):
#         raise ValueError(
#             f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
#         )

# class SchedulerType(ExplicitEnum):
#     LINEAR = "linear"
#     COSINE = "cosine"
#     COSINE_WITH_RESTARTS = "cosine_with_restarts"
#     POLYNOMIAL = "polynomial"
#     CONSTANT = "constant"
#     CONSTANT_WITH_WARMUP = "constant_with_warmup"

# TYPE_TO_SCHEDULER_FUNCTION = {
#     SchedulerType.LINEAR: get_linear_schedule_with_warmup,
#     SchedulerType.COSINE: get_cosine_schedule_with_warmup,
#     SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
#     SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
#     SchedulerType.CONSTANT: get_constant_schedule,
#     SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
# }


# def get_scheduler(
#     name: Union[str, SchedulerType],
#     optimizer: Optimizer,
#     num_warmup_steps: Optional[int] = None,
#     num_training_steps: Optional[int] = None,
# ):
#     """
#     Unified API to get any scheduler from its name.
#
#     Args:
#         name (:obj:`str` or `:obj:`SchedulerType`):
#             The name of the scheduler to use.
#         optimizer (:obj:`torch.optim.Optimizer`):
#             The optimizer that will be used during training.
#         num_warmup_steps (:obj:`int`, `optional`):
#             The number of warmup steps to do. This is not required by all schedulers (hence the argument being
#             optional), the function will raise an error if it's unset and the scheduler type requires it.
#         num_training_steps (:obj:`int`, `optional`):
#             The number of training steps to do. This is not required by all schedulers (hence the argument being
#             optional), the function will raise an error if it's unset and the scheduler type requires it.
#     """
#     name = SchedulerType(name)
#     schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
#     if name == SchedulerType.CONSTANT:
#         return schedule_func(optimizer)
#
#     # All other schedulers require `num_warmup_steps`
#     if num_warmup_steps is None:
#         raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")
#
#     if name == SchedulerType.CONSTANT_WITH_WARMUP:
#         return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)
#
#     # All other schedulers require `num_training_steps`
#     if num_training_steps is None:
#         raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")
#
#     return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


# class Adam(Optimizer):
#     """Implements BERT version of Adam algorithm with weight decay fix.
#     Params:
#         lr: learning rate
#         warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
#         t_total: total number of training steps for the learning
#             rate schedule, -1  means constant learning rate. Default: -1
#         schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
#         b1: Adams b1. Default: 0.9
#         b2: Adams b2. Default: 0.999
#         e: Adams epsilon. Default: 1e-6
#         weight_decay: Weight decay. Default: 0.01
#         max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
#     """
#
#     def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.01, max_grad_norm=1.0):
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
#         if schedule not in SCHEDULES:
#             raise ValueError("Invalid schedule parameter: {}".format(schedule))
#         if not 0.0 <= warmup < 1.0 and not warmup == -1:
#             raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
#         if not 0.0 <= b1 < 1.0:
#             raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
#         if not 0.0 <= b2 < 1.0:
#             raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
#         if not eps >= 0.0:
#             raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
#         defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
#                         b1=b1, b2=b2, e=eps, weight_decay=weight_decay,
#                         max_grad_norm=max_grad_norm)
#         super(Adam, self).__init__(params, defaults)
#
#     def get_lr(self):
#         lr = []
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 if len(state) == 0:
#                     return [0]
#                 if group['t_total'] != -1:
#                     schedule_fct = SCHEDULES[group['schedule']]
#                     lr_scheduled = group['lr'] * schedule_fct(
#                         state['step']/group['t_total'], group['warmup'])
#                 else:
#                     lr_scheduled = group['lr']
#                 lr.append(lr_scheduled)
#         return lr
#
#     def step(self, closure=None):
#         """Performs a single optimization step.
#
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
#                 if group['t_total'] != -1:
#                     schedule_fct = SCHEDULES[group['schedule']]
#                     lr_scheduled = group['lr'] * schedule_fct(
#                         state['step']/group['t_total'], group['warmup'])
#                 else:
#                     lr_scheduled = group['lr']
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