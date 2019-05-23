
"""
    code for the training model
"""

import torch
import torch.nn as nn
from operators import *
from genotypes import *
from utils import *

class Cell(nn.Module): 
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, weight):
        super(Cell, self).__init__()
        if reduction_prev:
            self.pre0 = FactorizedReduce(C_prev_prev, C, affine=True, track_stats=True)
        else:
            self.pre0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0,affine=True, track_stats=True)
        self.pre1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True, track_stats=True)

        self.C = C
        self.steps = steps
        self.reduction = reduction
        self.multiplier = multiplier
        self.arch = self._parse_arch_(weight)
        self._compile_(self.arch)

    def _parse_arch_(self, weight):
        s_n, e_n = 0, 0
        arch = []
        c_op = len(PRIMITIVES)
        i_op_none = PRIMITIVES.index('none')
        for i_b in range(self.steps):
            s_n, e_n = e_n, e_n + i_b + 2
            W = weight[s_n:e_n].copy()
            i_ns = sorted(range(i_b+2), key=lambda i_n : -max(W[i_n][i_op] for i_op in range(c_op) if i_op != i_op_none))[:2]
            for i_n in i_ns:
                i_op_best = None
                for i_op in range(c_op):
                    if i_op == i_op_none:
                        continue
                    if i_op_best is None or W[i_n][i_op] > W[i_n][i_op_best]:
                        i_op_best = i_op
                arch.append((PRIMITIVES[i_op_best], i_n))
        return arch

    def _compile_(self, arch):
        self._ops = nn.ModuleList()
        for (op_name, i_n) in arch:
            stride = 2 if self.reduction and i_n < 2 else 1
            self._ops.append(OPS[op_name](self.C, stride, True, True))
    
    def forward(self, s0, s1, drop_prob):
        s0, s1 = self.pre0(s0), self.pre1(s1)
        stats = [s0, s1]
        for i_b in range(self.steps):
            op_a, op_b = self._ops[i_b * 2], self._ops[i_b * 2 + 1]
            i_na, i_nb = self.arch[i_b * 2][1], self.arch[i_b * 2 + 1][1]
            i_a, i_b = stats[i_na], stats[i_nb]
            i_a, i_b = op_a(i_a), op_b(i_b)
            if self.training and drop_prob > 0.:
                if not isinstance(op_a, Identity):
                    i_a = drop_path(i_a, drop_prob)
                if not isinstance(op_b, Identity):
                    i_b = drop_path(i_b, drop_prob)
            stats.append(i_a+i_b)
        return torch.cat(stats[-self.multiplier:], 1)

class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x

class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3,auxiliary=True, arch=None, drop_path_prob=0.3):
        super(NetworkCIFAR, self).__init__()
        self.layers = layers
        self.auxiliary = auxiliary
        self.drop_path_prob = drop_path_prob

        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
                weight = arch['reduce']
            else:
                reduction = False
                weight = arch['normal']
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, weight)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
            if i == 2*layers//3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2*self.layers//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits, logits_aux