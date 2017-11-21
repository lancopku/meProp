'''
Define the new module that using meProp
Both meProp and unified meProp are supported
'''
import math
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter

from functions import linear, linearUnified


class Linear(nn.Module):
    '''
    A linear module (layer without activation) with meprop
    The initialization of w and b is the same with the default linear module.
    '''

    def __init__(self, in_, out_, k, unified=False):
        super(Linear, self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.k = k
        self.unified = unified

        self.w = Parameter(torch.Tensor(self.in_, self.out_))
        self.b = Parameter(torch.Tensor(self.out_))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.unified:
            return linearUnified(self.k)(x, self.w, self.b)
        else:
            return linear(self.k)(x, self.w, self.b)

    def __repr__(self):
        return '{} ({} -> {} <- {}{})'.format(self.__class__.__name__,
                                              self.in_, self.out_, 'unified'
                                              if self.unified else '', self.k)
