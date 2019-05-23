
#
# Date:    2019_05_11
# Author:  zhangxiong(1025679612@qq.com)
# Purpose: operators for differentiable neural architecture search
#

from operators import *
from genotypes import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class Node(nn.Module):
    def __init__(self, C, stride):
        super(Node, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False, track_running_stats=False))
            self._ops.append(op)
    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction  = reduction
        self.steps      = steps
        self.multiplier = multiplier

        if reduction_prev:
            self.pre0 = FactorizedReduce(C_prev_prev, C, affine=False, track_stats=False)
        else:
            self.pre0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0,affine=False, track_stats=False)
        self.pre1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False, track_stats=False)

        self._ops = nn.ModuleList()
        for i_b in range(self.steps):
            for i_n in range(2 + i_b):
                stride = 2 if reduction and i_n < 2 else 1
                self._ops.append(Node(C, stride))

    def forward(self, s0, s1, weights):
        stats = [self.pre0(s0), self.pre1(s1)]
        s_n, e_n = 0, 0
        for i_b in range(self.steps):
            s_n, e_n = e_n, e_n+i_b+2    
            stats.append(sum(self._ops[i_n](stats[i_n-s_n], weights[i_n]) for i_n in range(s_n, e_n)))

        return torch.cat(stats[-self.multiplier:], dim=1) #concat the last multiplier blocks's output

class Network(nn.Module):
    def __init__(self, C, num_classes, layer, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self.C = C
        self.num_classes = num_classes
        self.steps = steps
        self.multiplier = multiplier
        self.layer = layer
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
        self.cells = nn.ModuleList()
        for i_c in range(layer):
            if i_c in [layer//3 * 2, layer//3]:
                reduction = True
                C_curr = C_curr * 2
            else:
                reduction = False
            self.cells.append(Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev))
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier*C_curr
        
        self._initialize_alphas()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
    
    def forward(self, x):
        s0 = s1 = self.stem(x)
        w_alpha_normal    = F.softmax(self.w_alpha_normal,    1)
        w_alpha_reduction = F.softmax(self.w_alpha_reduction, 1)
        for i_c, cell in enumerate(self.cells):
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1, w_alpha_reduction)
            else:
                s0, s1 = s1, cell(s0, s1, w_alpha_normal)
        out = self.global_pooling(s1)
        return self.classifier(out.view(out.shape[0], -1))

    def _initialize_alphas(self):
        c_op   = len(PRIMITIVES)
        c_node = sum(i_b + 2 for i_b in range(self.steps))
        self.register_parameter('w_alpha_normal',    nn.Parameter(torch.rand((c_node, c_op))))
        self.register_parameter('w_alpha_reduction', nn.Parameter(torch.rand((c_node, c_op))))

if __name__ == '__main__':
    model = Network(4, 10, 8, 4, 4, 3).cuda()