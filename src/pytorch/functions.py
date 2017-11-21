'''
Define new functional operations that using meProp
Both meProp and unified meProp are supported
'''
import torch
from torch.autograd import Function


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

        if self.k > 0 and self.k < w.size(1):  # backprop with top-k selection
            _, inds = dy.abs().sum(0).topk(
                self.k)  # get top-k across examples in magnitude
            inds = inds.view(-1)  # flat
            pdy = dy.index_select(
                -1, inds
            )  # get the top-k values (k column) from dy and form a smaller dy matrix

            # compute the gradients of x, w, and b, using the smaller dy matrix
            if self.needs_input_grad[0]:
                dx = torch.mm(pdy, w.index_select(-1, inds).t_())
            if self.needs_input_grad[1]:
                dw = w.new(w.size()).zero_().index_copy_(
                    -1, inds, torch.mm(x.t(), pdy))
            if self.needs_input_grad[2]:
                db = torch.mv(dy.t(), self.add_buffer)
        else:  # backprop without top-k selection
            if self.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if self.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)
            if self.needs_input_grad[2]:
                db = torch.mv(dy.t(), self.add_buffer)

        return dx, dw, db


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

        if self.k > 0 and self.k < w.size(1):  # backprop with top-k selection
            _, indices = dy.abs().topk(self.k)
            if self.sparse:  # using sparse matrix multiplication
                values = dy.gather(-1, indices).view(-1)
                row_indices = torch.arange(
                    0, dy.size()[0]).long().cuda().unsqueeze_(-1).repeat(
                        1, self.k)
                indices = torch.stack([row_indices.view(-1), indices.view(-1)])
                pdy = torch.cuda.sparse.FloatTensor(indices, values, dy.size())
                if self.needs_input_grad[0]:
                    dx = torch.dsmm(pdy, w.t())
                if self.needs_input_grad[1]:
                    dw = torch.dsmm(pdy.t(), x).t()
            else:
                pdy = torch.cuda.FloatTensor(dy.size()).zero_().scatter_(
                    -1, indices, dy.gather(-1, indices))
                if self.needs_input_grad[0]:
                    dx = torch.mm(pdy, w.t())
                if self.needs_input_grad[1]:
                    dw = torch.mm(x.t(), pdy)
        else:  # backprop without top-k selection
            if self.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if self.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)

        if self.needs_input_grad[2]:
            db = torch.mv(dy.t(), self.add_buffer)

        return dx, dw, db
