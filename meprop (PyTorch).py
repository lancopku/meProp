# coding: utf-8

# In[1]:

'''
Python 3 code of meProp
MNIST Task
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import math
import time

import numpy

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import Function

from statistics import mean
from collections import OrderedDict


# In[2]:

class PartDataset(torch.utils.data.Dataset):
    '''
    Partial Dataset: 
        Extract the examples from the given dataset, 
        starting from the offset. 
        Stop if reach the length.
    '''
    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        super(PartDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.dataset[i+self.offset]

def get_mnist(datapath='./data/', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    trn = datasets.MNIST(
            datapath,
            train=True,
            download=download,
            transform=transforms.ToTensor()
        )
    dev = PartDataset(trn, 0, 5000)
    trnn = PartDataset(trn, 5000, 55000)
    tst = datasets.MNIST(
        datapath,
        train=False,
        transform=transforms.ToTensor()
        )
    return trnn, dev, tst

def get_artificial_dataset(nsample, ninfeature, noutfeature):
    '''
    Generate a synthetic dataset.
    '''
    data = torch.randn(nsample, ninfeature).cuda()
    target = torch.LongTensor(numpy.random.randint(noutfeature, size=(nsample, 1))).cuda()
    return torch.utils.data.TensorDataset(data, target)


# In[3]:

class linearUnified(Function):
    '''
    linear function with meProp, unified top-k across minibatch
    y = f(x, w, b) = xw + b
    '''

    def __init__(self, k):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linearUnified, self).__init__()
        self.k = k

    def forward(self, x, w, b):
        '''
        forward propagation
        x should be of size [minibatch, input feature]
        w should be of size [input feature, output feature]
        b should be of size [output feature]
        
        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [output feature, input feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        self.save_for_backward(x, w, b)
        y = x.new(x.size(0), w.size(1))
        y.addmm_(0, 1, x, w)
        self.add_buffer = x.new(x.size(0)).fill_(1)
        y.addr_(self.add_buffer, b)
        return y

    def backward(self, dy):
        '''
        backprop with meprop
        if k is invalid (<=0 or > output feature), no top-k selection is applied
        '''
        x, w, b = self.saved_tensors
        dx = dw = db = None


        if self.k>0 and self.k<w.size(1): # backprop with top-k selection
            _, inds = dy.abs().sum(0).topk(self.k) # get top-k across examples in magnitude
            inds = inds.view(-1) # flat
            pdy = dy.index_select(-1, inds) # get the top-k values (k column) from dy and form a smaller dy matrix

            # compute the gradients of x, w, and b, using the smaller dy matrix
            if self.needs_input_grad[0]:
                dx = torch.mm(pdy, w.index_select(-1, inds).t_())
            if self.needs_input_grad[1]:
                dw = w.new(w.size()).zero_().index_copy_(-1, inds, torch.mm(x.t(), pdy))
            if self.needs_input_grad[2]:
                db = torch.mv(dy.t(), self.add_buffer)
        else: # backprop without top-k selection
            if self.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if self.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)
            if self.needs_input_grad[2]:
                db = torch.mv(dy.t(), self.add_buffer)

        return dx, dw, db


# In[4]:

class linear(Function):
    '''
    linear function with meProp, top-k selection with respect to each example in minibatch
    y = f(x, w, b) = xw + b
    '''

    def __init__(self, k, sparse=True):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linear, self).__init__()
        self.k = k
        self.sparse = sparse

    def forward(self, x, w, b):
        '''
        forward propagation
        x should be of size [minibatch, input feature]
        w should be of size [input feature, output feature]
        b should be of size [output feature]
        
        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [output feature, input feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        self.save_for_backward(x, w, b)
        y = x.new(x.size(0), w.size(1))
        y.addmm_(0, 1, x, w)
        self.add_buffer = x.new(x.size(0)).fill_(1)
        y.addr_(self.add_buffer, b)
        return y

    def backward(self, dy):
        '''
        backprop with meprop
        if k is invalid (<=0 or > output feature), no top-k selection is applied
        '''
        x, w, b = self.saved_tensors
        dx = dw = db = None


        if self.k>0 and self.k<w.size(1): # backprop with top-k selection
            _, indices = dy.abs().topk(self.k)
            if self.sparse: # using sparse matrix multiplication
                values = dy.gather(-1, indices).view(-1)
                row_indices = torch.arange(0, dy.size()[0]).long().cuda().unsqueeze_(-1).repeat(1, self.k)
                indices = torch.stack([row_indices.view(-1), indices.view(-1)])
                pdy = torch.cuda.sparse.FloatTensor(indices, values, dy.size())
                if self.needs_input_grad[0]:
                    dx = torch.dsmm(pdy, w.t())
                if self.needs_input_grad[1]:
                    dw = torch.dsmm(pdy.t(), x).t()
            else:
                pdy = torch.cuda.FloatTensor(dy.size()).zero_().scatter_(-1, indices, dy.gather(-1, indices))
                if self.needs_input_grad[0]:
                    dx = torch.mm(pdy, w.t())
                if self.needs_input_grad[1]:
                    dw = torch.mm(x.t(), pdy)
        else: # backprop without top-k selection
            if self.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if self.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)

        if self.needs_input_grad[2]:
            db = torch.mv(dy.t(), self.add_buffer)

        return dx, dw, db


# In[5]:

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
        return '{} ({} -> {} <- {}{})'.format(self.__class__.__name__, self.in_, self.out_, 'unified' if self.unified else '', self.k)



# In[6]:

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
                d['linear'+str(i)] = Linear(784, hidden, k, self.unified)
                d['relu'+str(i)] = nn.ReLU()
                if dropout:
                    d['dropout'+str(i)] = nn.Dropout(p=dropout)
            elif i==layer-1:
                d['linear'+str(i)] = Linear(hidden, 10, 0, self.unified)
            else:
                d['linear'+str(i)] = Linear(hidden, hidden, k, self.unified)
                d['relu'+str(i)] = nn.ReLU()
                if dropout:
                    d['dropout'+str(i)] = nn.Dropout(p=dropout)
        return d

    def forward(self, x):
        return F.log_softmax(self.model(x.view(-1, 784)))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, type(Linear)):
                m.reset_parameters()


# In[7]:

class TestGroup(object):
    '''
    A network and k in meporp form a test group.
    Test groups differ in minibatch size, hidden features, layer number and dropout rate.
    '''
    def __init__(self, trnset, mb, hidden, layer, dropout, unified, devset=None, tstset=None, cudatensor=False, file=sys.stdout):
        self.mb = mb
        self.hidden = hidden
        self.layer = layer
        self.dropout = dropout
        self.file = file
        self.trnset = trnset
        self.unified = unified

        if cudatensor: # dataset is on GPU
            self.trainloader = torch.utils.data.DataLoader(trnset, batch_size=mb, num_workers=0)
            if tstset:
                self.testloader = torch.utils.data.DataLoader(tstset, batch_size=mb, num_workers=0)
            else:
                self.testloader = None
        else: # dataset is on CPU, using prefetch and pinned memory to shorten the data transfer time
            self.trainloader = torch.utils.data.DataLoader(trnset, batch_size=mb, shuffle=True, num_workers=1, pin_memory=True)
            if devset:
                self.devloader = torch.utils.data.DataLoader(devset, batch_size=mb, shuffle=False, num_workers=1, pin_memory=True)
            else:
                self.devloader = None
            if tstset:
                self.testloader = torch.utils.data.DataLoader(tstset, batch_size=mb, shuffle=False, num_workers=1, pin_memory=True)
            else:
                self.testloader = None
        self.basettime = None
        self.basebtime = None

    def reset(self):
        '''
        Reinit the trainloader at the start of each run,
        so that the traning examples is in the same random order
        '''
        torch.manual_seed(12976)
        self.trainloader = torch.utils.data.DataLoader(self.trnset, batch_size=self.mb, shuffle=True, num_workers=1, pin_memory=True)

    def _train(self, model, opt):
        '''
        Train the given model using the given optimizer
        Record the time and loss
        '''
        model.train()
        ftime = 0
        btime = 0
        utime = 0
        tloss = 0
        for bid, (data, target) in enumerate(self.trainloader):
            data, target = Variable(data).cuda(), Variable(target.view(-1)).cuda()
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            opt.zero_grad()
            end.record()
            end.synchronize()
            utime += start.elapsed_time(end)

            start.record()
            output = model(data)
            loss = F.nll_loss(output, target)
            end.record()
            end.synchronize()
            ftime += start.elapsed_time(end)

            start.record()
            loss.backward()
            end.record()
            end.synchronize()
            btime += start.elapsed_time(end)

            start.record()
            opt.step()
            end.record()
            end.synchronize()
            utime += start.elapsed_time(end)

            tloss += loss.data[0]
        tloss /= len(self.trainloader)
        return tloss, ftime, btime, utime

    def _evaluate(self, model, loader, name='test'):
        '''
        Use the given model to classify the examples in the given data loader
        Record the loss and accuracy.
        '''
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in loader:
            data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target).data[0]
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(loader)  # loss function already averages over batch size
        print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.
              format(name, test_loss, correct,
                     len(loader.dataset), 100. * correct / len(loader.dataset)), file=self.file, flush=True)
        return 100. * correct / len(loader.dataset)

    def run(self, k, epoch=20):
        '''
        Run a training loop.
        '''

        print('mbsize: {}, hidden size: {}, layer: {}, dropout: {}, k: {}'.format(self.mb, self.hidden, self.layer, self.dropout, k), file=self.file)
        # Init the model, the optimizer and some structures for logging
        self.reset()

        model = NetLayer(self.hidden, k, self.layer, self.dropout, self.unified)
        #print(model)
        model.reset_parameters()
        model.cuda()

        opt = optim.Adam(model.parameters())

        acc = 0 # best dev. acc.
        accc = 0 # test acc. at the time of best dev. acc.
        e = -1 # best dev iteration/epoch

        times = []
        losses = []
        ftime = []
        btime = []
        utime = []

        # training loop
        for t in range(epoch):
            print('{}：'.format(t), end='', file=self.file, flush=True)
            # train
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            loss, ft, bt, ut = self._train(model, opt)
            end.record()
            end.synchronize()
            ttime = start.elapsed_time(end)

            times.append(ttime)
            losses.append(loss)
            ftime.append(ft)
            btime.append(bt)
            utime.append(ut)
            # predict
            curacc = self._evaluate(model, self.devloader, 'dev')
            if curacc > acc:
                e = t
                acc = curacc
                accc = self._evaluate(model, self.testloader, '    test')
        etime = [sum(t) for t in zip(ftime, btime, utime)]
        print('${:.2f}|{:.2f} at {}'.format(acc, accc, e), file=self.file, flush=True)
        print('',file=self.file)


    def _stat(self, name, t, agg=mean):
        return '{:<5}:\t{:8.3f}; {}'.format(name, agg(t), ', '.join(['{:8.2f}'.format(x) for x in t]))


# In[8]:

# a simple use example (not unified)
group = TestGroup(trn, 32, 512, 3, 0.1, False, devset=dev, tstset=tst, file=sys.stdout)

group.run(0, 20)
group.run(30, 20)

# results may be different at each run
# mbsize: 32, hidden size: 512, layer: 3, dropout: 0.1, k: 0
# 0：dev set: Average loss: 0.1202, Accuracy: 4826/5000 (96.52%)
#     test set: Average loss: 0.1272, Accuracy: 9616/10000 (96.16%)
# 1：dev set: Average loss: 0.1047, Accuracy: 4832/5000 (96.64%)
#     test set: Average loss: 0.1131, Accuracy: 9658/10000 (96.58%)
# 2：dev set: Average loss: 0.0786, Accuracy: 4889/5000 (97.78%)
#     test set: Average loss: 0.0834, Accuracy: 9749/10000 (97.49%)
# 3：dev set: Average loss: 0.0967, Accuracy: 4875/5000 (97.50%)
# 4：dev set: Average loss: 0.0734, Accuracy: 4907/5000 (98.14%)
#     test set: Average loss: 0.0818, Accuracy: 9790/10000 (97.90%)
# 5：dev set: Average loss: 0.0848, Accuracy: 4894/5000 (97.88%)
# 6：dev set: Average loss: 0.0750, Accuracy: 4904/5000 (98.08%)
# 7：dev set: Average loss: 0.0882, Accuracy: 4897/5000 (97.94%)
# 8：dev set: Average loss: 0.0978, Accuracy: 4896/5000 (97.92%)
# 9：dev set: Average loss: 0.0828, Accuracy: 4910/5000 (98.20%)
#     test set: Average loss: 0.0973, Accuracy: 9797/10000 (97.97%)
# 10：dev set: Average loss: 0.1004, Accuracy: 4901/5000 (98.02%)
# 11：dev set: Average loss: 0.0813, Accuracy: 4914/5000 (98.28%)
#     test set: Average loss: 0.0917, Accuracy: 9795/10000 (97.95%)
# 12：dev set: Average loss: 0.0880, Accuracy: 4912/5000 (98.24%)
# 13：dev set: Average loss: 0.1106, Accuracy: 4910/5000 (98.20%)
# 14：dev set: Average loss: 0.0981, Accuracy: 4921/5000 (98.42%)
#     test set: Average loss: 0.1065, Accuracy: 9824/10000 (98.24%)
# 15：dev set: Average loss: 0.1044, Accuracy: 4914/5000 (98.28%)
# 16：dev set: Average loss: 0.1235, Accuracy: 4904/5000 (98.08%)
# 17：dev set: Average loss: 0.1202, Accuracy: 4910/5000 (98.20%)
# 18：dev set: Average loss: 0.1021, Accuracy: 4926/5000 (98.52%)
#     test set: Average loss: 0.1167, Accuracy: 9800/10000 (98.00%)
# 19：dev set: Average loss: 0.1490, Accuracy: 4910/5000 (98.20%)
# $98.52|98.00 at 18

# mbsize: 32, hidden size: 512, layer: 3, dropout: 0.1, k: 30
# 0：dev set: Average loss: 0.3283, Accuracy: 4754/5000 (95.08%)
#     test set: Average loss: 0.3611, Accuracy: 9480/10000 (94.80%)
# 1：dev set: Average loss: 0.1762, Accuracy: 4804/5000 (96.08%)
#     test set: Average loss: 0.1837, Accuracy: 9592/10000 (95.92%)
# 2：dev set: Average loss: 0.1192, Accuracy: 4871/5000 (97.42%)
#     test set: Average loss: 0.1231, Accuracy: 9715/10000 (97.15%)
# 3：dev set: Average loss: 0.0966, Accuracy: 4875/5000 (97.50%)
#     test set: Average loss: 0.0978, Accuracy: 9745/10000 (97.45%)
# 4：dev set: Average loss: 0.1027, Accuracy: 4880/5000 (97.60%)
#     test set: Average loss: 0.0846, Accuracy: 9774/10000 (97.74%)
# 5：dev set: Average loss: 0.1052, Accuracy: 4865/5000 (97.30%)
# 6：dev set: Average loss: 0.0888, Accuracy: 4889/5000 (97.78%)
#     test set: Average loss: 0.0936, Accuracy: 9773/10000 (97.73%)
# 7：dev set: Average loss: 0.0840, Accuracy: 4894/5000 (97.88%)
#     test set: Average loss: 0.0890, Accuracy: 9797/10000 (97.97%)
# 8：dev set: Average loss: 0.0845, Accuracy: 4894/5000 (97.88%)
# 9：dev set: Average loss: 0.0896, Accuracy: 4893/5000 (97.86%)
# 10：dev set: Average loss: 0.0929, Accuracy: 4902/5000 (98.04%)
#     test set: Average loss: 0.1005, Accuracy: 9785/10000 (97.85%)
# 11：dev set: Average loss: 0.0903, Accuracy: 4906/5000 (98.12%)
#     test set: Average loss: 0.0950, Accuracy: 9792/10000 (97.92%)
# 12：dev set: Average loss: 0.1019, Accuracy: 4906/5000 (98.12%)
# 13：dev set: Average loss: 0.0870, Accuracy: 4919/5000 (98.38%)
#     test set: Average loss: 0.0894, Accuracy: 9823/10000 (98.23%)
# 14：dev set: Average loss: 0.1003, Accuracy: 4909/5000 (98.18%)
# 15：dev set: Average loss: 0.1162, Accuracy: 4910/5000 (98.20%)
# 16：dev set: Average loss: 0.1031, Accuracy: 4920/5000 (98.40%)
#     test set: Average loss: 0.1001, Accuracy: 9828/10000 (98.28%)
# 17：dev set: Average loss: 0.0994, Accuracy: 4913/5000 (98.26%)
# 18：dev set: Average loss: 0.0969, Accuracy: 4905/5000 (98.10%)
# 19：dev set: Average loss: 0.1095, Accuracy: 4920/5000 (98.40%)
# $98.40|98.28 at 16

# In[9]:

# a simple use example (unified)
# change the sys.stdout to a file object to write the results to the file
trn, dev, tst = get_mnist()
group = TestGroup(trn, 50, 500, 3, 0.1, True, devset=dev, tstset=tst, file=sys.stdout)

group.run(0, 20)
group.run(30, 20)

# results may be different at each run
# mbsize: 50, hidden size: 500, layer: 3, dropout: 0.1, k: 0
# 0：dev set: Average loss: 0.1043, Accuracy: 4843/5000 (96.86%)
#     test set: Average loss: 0.1163, Accuracy: 9655/10000 (96.55%)
# 1：dev set: Average loss: 0.0789, Accuracy: 4892/5000 (97.84%)
#     test set: Average loss: 0.0792, Accuracy: 9766/10000 (97.66%)
# 2：dev set: Average loss: 0.0818, Accuracy: 4875/5000 (97.50%)
# 3：dev set: Average loss: 0.0823, Accuracy: 4880/5000 (97.60%)
# 4：dev set: Average loss: 0.0869, Accuracy: 4888/5000 (97.76%)
# 5：dev set: Average loss: 0.0810, Accuracy: 4904/5000 (98.08%)
#     test set: Average loss: 0.0711, Accuracy: 9807/10000 (98.07%)
# 6：dev set: Average loss: 0.0752, Accuracy: 4903/5000 (98.06%)
# 7：dev set: Average loss: 0.0805, Accuracy: 4907/5000 (98.14%)
#     test set: Average loss: 0.0833, Accuracy: 9799/10000 (97.99%)
# 8：dev set: Average loss: 0.1105, Accuracy: 4876/5000 (97.52%)
# 9：dev set: Average loss: 0.0913, Accuracy: 4901/5000 (98.02%)
# 10：dev set: Average loss: 0.0800, Accuracy: 4915/5000 (98.30%)
#     test set: Average loss: 0.0832, Accuracy: 9830/10000 (98.30%)
# 11：dev set: Average loss: 0.0909, Accuracy: 4913/5000 (98.26%)
# 12：dev set: Average loss: 0.0908, Accuracy: 4907/5000 (98.14%)
# 13：dev set: Average loss: 0.0753, Accuracy: 4920/5000 (98.40%)
#     test set: Average loss: 0.0902, Accuracy: 9803/10000 (98.03%)
# 14：dev set: Average loss: 0.0947, Accuracy: 4918/5000 (98.36%)
# 15：dev set: Average loss: 0.0800, Accuracy: 4918/5000 (98.36%)
# 16：dev set: Average loss: 0.0822, Accuracy: 4915/5000 (98.30%)
# 17：dev set: Average loss: 0.1059, Accuracy: 4916/5000 (98.32%)
# 18：dev set: Average loss: 0.1128, Accuracy: 4914/5000 (98.28%)
# 19：dev set: Average loss: 0.0936, Accuracy: 4924/5000 (98.48%)
#     test set: Average loss: 0.1214, Accuracy: 9794/10000 (97.94%)
# $98.48|97.94 at 19

# mbsize: 50, hidden size: 500, layer: 3, dropout: 0.1, k: 30
# 0：dev set: Average loss: 0.1749, Accuracy: 4743/5000 (94.86%)
#     test set: Average loss: 0.1867, Accuracy: 9466/10000 (94.66%)
# 1：dev set: Average loss: 0.1180, Accuracy: 4828/5000 (96.56%)
#     test set: Average loss: 0.1219, Accuracy: 9638/10000 (96.38%)
# 2：dev set: Average loss: 0.1190, Accuracy: 4818/5000 (96.36%)
# 3：dev set: Average loss: 0.0901, Accuracy: 4868/5000 (97.36%)
#     test set: Average loss: 0.0984, Accuracy: 9697/10000 (96.97%)
# 4：dev set: Average loss: 0.0834, Accuracy: 4876/5000 (97.52%)
#     test set: Average loss: 0.0883, Accuracy: 9736/10000 (97.36%)
# 5：dev set: Average loss: 0.0849, Accuracy: 4866/5000 (97.32%)
# 6：dev set: Average loss: 0.0775, Accuracy: 4891/5000 (97.82%)
#     test set: Average loss: 0.0893, Accuracy: 9732/10000 (97.32%)
# 7：dev set: Average loss: 0.0767, Accuracy: 4893/5000 (97.86%)
#     test set: Average loss: 0.0777, Accuracy: 9770/10000 (97.70%)
# 8：dev set: Average loss: 0.0788, Accuracy: 4881/5000 (97.62%)
# 9：dev set: Average loss: 0.0816, Accuracy: 4894/5000 (97.88%)
#     test set: Average loss: 0.0840, Accuracy: 9756/10000 (97.56%)
# 10：dev set: Average loss: 0.0820, Accuracy: 4884/5000 (97.68%)
# 11：dev set: Average loss: 0.0776, Accuracy: 4887/5000 (97.74%)
# 12：dev set: Average loss: 0.0779, Accuracy: 4896/5000 (97.92%)
#     test set: Average loss: 0.0740, Accuracy: 9791/10000 (97.91%)
# 13：dev set: Average loss: 0.0811, Accuracy: 4891/5000 (97.82%)
# 14：dev set: Average loss: 0.0817, Accuracy: 4904/5000 (98.08%)
#     test set: Average loss: 0.0831, Accuracy: 9783/10000 (97.83%)
# 15：dev set: Average loss: 0.0828, Accuracy: 4895/5000 (97.90%)
# 16：dev set: Average loss: 0.0836, Accuracy: 4897/5000 (97.94%)
# 17：dev set: Average loss: 0.0774, Accuracy: 4904/5000 (98.08%)
# 18：dev set: Average loss: 0.0783, Accuracy: 4914/5000 (98.28%)
#     test set: Average loss: 0.0707, Accuracy: 9804/10000 (98.04%)
# 19：dev set: Average loss: 0.0873, Accuracy: 4901/5000 (98.02%)
# $98.28|98.04 at 18
