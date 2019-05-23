
#
# Date:    2019_05_11
# Author:  zhangxiong(1025679612@qq.com)
# Purpose: operators for differentiable neural architecture search
#

import torch
import torch.nn as nn

class OP(nn.Module):
    def __init__(self):
        super(OP, self).__init__()

    def forward(self, *x):
        print("pure virtual operator is not callable.")
        assert 0

class ReLUConvBN(OP):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, track_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_stats)
        )

    def forward(self, x):
        return self.op(x)

class Zero(OP):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        else:
            return x[:, :, ::self.stride, ::self.stride].mul(0.0)

class Identity(OP):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

#depth-wise conv
class SepConv(OP):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True,track_stats=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,stride=stride,padding=padding,groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(C_in, affine=affine,track_running_stats=track_stats),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,stride=1,padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(C_out, affine=affine,track_running_stats=track_stats)
        )
    
    def forward(self, x):
        return self.op(x)

class TwoPassConv(OP):
    def __init__(self, C_in, C_out, kernel_size, stride, affine=True, track_stats=True):
        super(TwoPassConv, self).__init__()
        padding = stride // 2
        if C_in != C_out:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size),stride=(1,stride),padding=(0,padding),bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=(kernel_size, 1),stride=(stride,1),padding=(padding,0),bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1,stride=1,bias=False),
                nn.BatchNorm2d(C_out, affine=affine,track_running_stats=track_stats)
            )
        else:
            C = C_in
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C, C, kernel_size=(1, kernel_size),stride=(1,stride),padding=(0,padding),bias=False),
                nn.Conv2d(C, C, kernel_size=(kernel_size, 1),stride=(stride,1),padding=(padding,0),bias=False),
                nn.BatchNorm2d(C, affine=affine,track_running_stats=track_stats)
            )

    def forward(self, x):
        return self.op(x)

class DilConv(OP):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, track_stats=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_stats),
        )

    def forward(self, x):
        return self.op(x)

class FactorizedReduce(OP):
    def __init__(self, C_in, C_out, affine=True, track_stats=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_stats)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out
    
OPS = {
    'none'         : lambda C, stride, affine, track_stats : Zero(stride),
    'avg_pool_3x3' : lambda C, stride, affine, track_stats : nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3' : lambda C, stride, affine, track_stats : nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect' : lambda C, stride, affine, track_stats : Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3' : lambda C, stride, affine, track_stats : SepConv(C, C, 3, stride, 1, affine, track_stats),
    'sep_conv_5x5' : lambda C, stride, affine, track_stats : SepConv(C, C, 5, stride, 2, affine, track_stats),
    'sep_conv_7x7' : lambda C, stride, affine, track_stats : SepConv(C, C, 7, stride, 3, affine, track_stats),
    'dil_conv_3x3' : lambda C, stride, affine, track_stats : DilConv(C, C, 3, stride, 2, 2, affine, track_stats),
    'dil_conv_5x5' : lambda C, stride, affine, track_stats : DilConv(C, C, 5, stride, 4, 2, affine, track_stats),
    'conv_7x1_1x7' : lambda C, stride, affine, track_stats : TwoPassConv(C, C, 7, stride, affine, track_stats)
}

if __name__ == '__main__':
    model = TwoPassConv(3, 64, 3, 2)
    inputs = torch.rand((1, 3, 64,64))
    print(model(inputs).shape)