import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        _, _, h, w = self.weight.size()
        self.mask[:, :, :h//2, :] = 1
        if mask_type == 'A':
            self.mask[:, :, h//2, :w//2] = 1
        else:
            self.mask[:, :, h//2, :w//2+1] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)