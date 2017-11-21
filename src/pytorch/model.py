'''
The MLP model for MNIST
'''

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Linear


class NetLayer(nn.Module):
    '''
    A complete network(MLP) for MNSIT classification.
    
    Input feature is 28*28=784
    Output feature is 10
    Hidden features are of hidden size
    
    Activation is ReLU
    '''

    def __init__(self, hidden, k, layer, dropout=None, unified=False):
        super(NetLayer, self).__init__()
        self.k = k
        self.layer = layer
        self.dropout = dropout
        self.unified = unified
        self.model = nn.Sequential(self._create(hidden, k, layer, dropout))

    def _create(self, hidden, k, layer, dropout=None):
        if layer == 1:
            return OrderedDict([Linear(784, 10, 0)])
        d = OrderedDict()
        for i in range(layer):
            if i == 0:
                d['linear' + str(i)] = Linear(784, hidden, k, self.unified)
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
            elif i == layer - 1:
                d['linear' + str(i)] = Linear(hidden, 10, 0, self.unified)
            else:
                d['linear' + str(i)] = Linear(hidden, hidden, k, self.unified)
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
        return d

    def forward(self, x):
        return F.log_softmax(self.model(x.view(-1, 784)))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, type(Linear)):
                m.reset_parameters()
