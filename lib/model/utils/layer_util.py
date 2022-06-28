import torch
import torch.nn as nn
import torch.nn.functional as F


class LargeSeparableConv2d(nn.Module):
  def __init__(self, c_in, dim_out = 490, kernel_size=15, bias=False, bn=False, setting='L'):
    super(LargeSeparableConv2d, self).__init__()
    
    # dim_out = 10 * 7 * 7    
    c_mid = 64 if setting == 'S' else 256

    self.din = c_in
    self.c_mid = c_mid
    self.c_out = dim_out
    self.k_width = (kernel_size, 1)
    self.k_height = (1, kernel_size)
    self.pad = 0
    self.bias = bias
    self.bn = bn

    self.block1_1 = nn.Conv2d(self.din, self.c_mid, self.k_width, 1, padding=self.pad, bias=self.bias)
    self.bn1_1 = nn.BatchNorm2d(self.c_mid)
    self.block1_2 = nn.Conv2d(self.c_mid, self.c_out, self.k_height, 1, padding=self.pad, bias=self.bias)
    self.bn1_2 = nn.BatchNorm2d(self.c_out)

    self.block2_1 = nn.Conv2d(self.din, self.c_mid, self.k_height, 1, padding=self.pad, bias=self.bias)
    self.bn2_1 = nn.BatchNorm2d(self.c_mid)
    self.block2_2 = nn.Conv2d(self.c_mid, self.c_out, self.k_width, 1, padding=self.pad, bias=self.bias)
    self.bn2_2 = nn.BatchNorm2d(self.c_out)

  def forward(self, x):
    x1 = self.block1_1(x)
    x1 = self.bn1_1(x1) if self.bn else x1
    x1 = self.block1_2(x1)
    x1 = self.bn1_2(x1) if self.bn else x1

    x2 = self.block2_1(x)
    x2 = self.bn2_1(x2) if self.bn else x2
    x2 = self.block2_2(x2)
    x2 = self.bn2_2(x2) if self.bn else x2

    return x1 + x2