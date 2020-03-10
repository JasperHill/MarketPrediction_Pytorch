from __future__ import absolute_import, division

import os
import math
import torch
import torch.nn             as nn
import torch.nn.functional  as F
import numpy                as np
import rnn_passes_cpp

from torch.autograd         import Function

class LSTM_Op_pass(Function):
    @staticmethod
    def forward(ctx, input, c_p, h_p, xOps, hOps):
        h, c, Xs = rnn_passes_cpp.LSTM_Op_forward(input.to(torch.double),
                                                  c_p.to(torch.double),
                                                  h_p.to(torch.double),
                                                  xOps.to(torch.double),
                                                  hOps.to(torch.double))

        ctx.save_for_backward(input, Xs, c, c_p, h_p, xOps, hOps)
        return h, c, Xs

    @staticmethod
    def backward(ctx, grad_output, grad_c, grad_Xs):
        ## grad_c and grad_Xs are zero because c and Xs from forward are not passed to the next layer

        input, Xs, c, c_p, h_p, xOps, hOps = ctx.saved_tensors
        
        grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps = rnn_passes_cpp.LSTM_Op_backward(input.to(torch.double),
                                                                                               Xs.to(torch.double),
                                                                                               c.to(torch.double),
                                                                                               c_p.to(torch.double),
                                                                                               h_p.to(torch.double),
                                                                                               xOps.to(torch.double),
                                                                                               hOps.to(torch.double),
                                                                                               grad_output.to(torch.double))

        # grad_c_p and grad_h_p are zero because they are not trainable parameters
        return grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps

# LSTM_Op is an LSTM layer whose kernels are true operators rather than Hadamard-like operators
# biases are not used
# multi-layer functionality is not currently supported
# bidirectionality is not currently supported

#todo: modify LSTM_Op to accept information regarding the time lapse between sequence elements
class LSTM_Op(nn.Module):
    def __init__(self, batch_size, input_channels, input_dim, output_channels, output_dim):
        super(LSTM_Op, self).__init__()
        self.input_channels = input_channels        
        self.input_dim = input_dim
        
        self.output_channels = output_channels
        self.output_dim = output_dim

        # operators act to the right -> Op*input
        # x operators map input sequences of length input_dim to sequences of length output_dim
        # h operators are square matrices of dimension hidden_size
        xOps = torch.zeros([4, self.output_channels, self.input_channels, self.output_dim, self.input_dim])
        hOps = torch.empty([4, self.output_channels, self.input_channels, self.output_dim, self.output_dim])
        hOps[:,:] = torch.eye(self.output_dim)
        
        # set x operators to identity in input_dim-dimensional space
        for i in range(min(self.output_dim, self.input_dim)):
            xOps[:,:,:,i,i] = 1

        self.xOps = nn.Parameter(xOps)
        self.hOps = nn.Parameter(hOps)
        

    def forward(self, input):
        batch_size = input.shape[0]
        self.c = torch.zeros([batch_size, self.input_channels, self.output_dim])
        self.h = torch.zeros([batch_size, self.input_channels, self.output_dim])
        
        self.h, self.c, _ = LSTM_Op_pass.apply(input, self.c, self.h, self.xOps, self.hOps)
        
        return self.h

