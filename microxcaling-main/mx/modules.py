# microxcaling-main/mx/modules.py
import torch.nn as nn
from .mx_ops import quantize_mx_op # ← 변경된 부분

class Conv2d(nn.Conv2d):
    def __init__(self, *args, mx_specs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mx_specs = mx_specs

    def forward(self, x):
        x = quantize_mx_op(x, self.mx_specs)
        out = super().forward(x)
        return quantize_mx_op(out, self.mx_specs)
