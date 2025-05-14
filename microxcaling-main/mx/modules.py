# microxcaling-main/mx/modules.py
import torch.nn as nn
from .ops import quantize_tensor

class Conv2d(nn.Conv2d):
    def __init__(self, *args, mx_specs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mx_specs = mx_specs

    def forward(self, x):
        x = quantize_tensor(x, self.mx_specs, mode="input")
        out = super().forward(x)
        return quantize_tensor(out, self.mx_specs, mode="output")
