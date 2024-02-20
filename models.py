"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.double)
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

class linear_transf(nn.Module):

    def __init__(self):
        super().__init__()
        # self.fc3 = nn.Linear(2,2, bias=True)
        self.fc3 = nn.Linear(2,2, bias=False)
        self.fc3.weight.data.mul_(0.1) # to remove

    def forward(self, x):
        x = self.fc3(x)
        return x