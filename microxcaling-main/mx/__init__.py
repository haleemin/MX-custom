"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

from .specs import MxSpecs
from .specs import add_mx_args, finalize_mx_specs
from .specs import get_mx_specs, get_backwards_mx_specs

from .quantize import quantize_bfloat

from .linear import Linear, linear
from .matmul import matmul
from .bmm import bmm

from .convolution import Conv1d, Conv2d, Conv3d
from .convolution import conv1d, conv2d, conv3d
from .transpose_convolution import ConvTranspose2d

from .rnn import LSTM

from .activations import sigmoid, tanh, relu, relu6, leaky_relu, silu, gelu
from .activations import Sigmoid, Tanh, ReLU, ReLU6, LeakyReLU, SiLU, GELU

from .adaptive_avg_pooling import adaptive_avg_pool2d, AdaptiveAvgPool2d

from .batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from .layernorm import LayerNorm, RMSNorm, layer_norm
from .groupnorm import GroupNorm

from .simd_ops import simd_add, simd_sub, simd_mul, simd_div, simd_split
from .simd_ops import simd_sqrt, simd_square, simd_exp, simd_log
from .simd_ops import simd_reduce_sum, simd_reduce_mean, simd_norm

from .modules import Conv2d
import torch.nn as _nn

# torch.nn.Conv2d → MX 양자화 Conv2d 로 교체
_nn.Conv2d = Conv2d


from .simd_ops import simd_add, simd_sub, simd_mul, simd_split
from .simd_ops import simd_sqrt, simd_square, simd_norm

from .softmax import Softmax, softmax
