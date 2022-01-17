import random
import numpy as np
import math
import os
import sys
# from functools import wraps
# import shutil
import torch
# import fnmatch
import torch.utils.checkpoint
from packaging import version
from torch import nn
# import json
# import tempfile
# from hashlib import sha256
# import boto3
# import requests
import time
import inspect
from datetime import timedelta
# from botocore.exceptions import ClientError
import logging
import threading

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

try:
    from pathlib import Path
    # get local model path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                   Path.home() / '.pytorch_pretrained_bert'))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                              os.path.join(os.path.expanduser("~"), '.pytorch_pretrained_bert'))
_lock = threading.Lock()
_default_logger = None  # 默认的logger
logger = logging.getLogger()


def get_logger(name=None, log_file=None, log_level=logging.INFO):
    """返回唯一的logger，支持console和文件"""
    # logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    if name is None:
        name = __name__.split(".")[0]
    global _default_logger

    with _lock:
        if _default_logger:
            return _default_logger
        _default_logger = logging.getLogger(name)
        _default_logger.setLevel(log_level)
        console_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        console_handler.flush = sys.stderr.flush
        console_handler.setFormatter(log_format)
        _default_logger.addHandler(console_handler)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            _default_logger.addHandler(file_handler)
    return _default_logger


def apply_chunking_to_forward(forward_fn, chunk_size: int, chunk_dim: int, *input_tensors):
    """ 将运算分割为多个chunk计算来节省显存 当chunk=0，不分块执行
    参数:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
        chunk_size : `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`): chunk的维度
        input_tensors :tuple(tensors)
    """
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"
    tensor_shape = input_tensors[0].shape[chunk_dim]
    # 每个tensor chunk的维度需相等
    assert all(input_tensor.shape[chunk_dim] == tensor_shape for input_tensor in input_tensors), "All input tenors have to be of the same shape"
    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input tensors are given")

    if chunk_size > 0:
        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk size {chunk_size}")
        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
        # 将每个tensor按chunk_dim分块
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # 执行forward到每个分块
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # 合并结果
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)

# 代码来自苏神的bert4keras https://github.com/bojone/bert4keras
def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        # 计算每个维度的最大长度
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def time_diff(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def set_seed(seed: int, set_random=True):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    """
    if set_random:
        random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if is_tf_available():
    #     tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True


def _gelu_python(x):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


if version.parse(torch.__version__) < version.parse("1.4"):
    gelu = _gelu_python
else:
    gelu = nn.functional.gelu

act2fn = {'relu': nn.functional.relu, 'sigmoid': torch.sigmoid, 'gelu': gelu}


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Returns list of all variables in the checkpoint
    init_vars = tf.train.list_variables(tf_path)

    names, arrays = [],  []
    for name, shape in init_vars:
        # logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
                for n in name):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            # if layer_{num} scope_name=num
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                # i.e. ['layer', '5', '']
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            # first find layer num
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (pointer.shape == array.shape), \
                f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model
