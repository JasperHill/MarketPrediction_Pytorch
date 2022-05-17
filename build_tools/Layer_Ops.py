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
        c, h, Xs = rnn_passes_cpp.LSTM_Op_forward(input.to(torch.double),
                                                  c_p.to(torch.double),
                                                  h_p.to(torch.double),
                                                  xOps.to(torch.double),
                                                  hOps.to(torch.double))

        ctx.save_for_backward(input, Xs, c, c_p, h_p, xOps, hOps)
        ## when backprop is called, gradients passed to backward calls
        ## will correspond directly to the outputs of forward alls
        return c, h, Xs

    @staticmethod
    def backward(ctx, grad_c, grad_h, grad_Xs):
        ## grad_c and grad_Xs are zero because c and Xs from forward are not passed to the next layer
        input, Xs, c, c_p, h_p, xOps, hOps = ctx.saved_tensors
        
        grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps = rnn_passes_cpp.LSTM_Op_backward(input.to(torch.double),
                                                                                               Xs.to(torch.double),
                                                                                               c.to(torch.double),
                                                                                               c_p.to(torch.double),
                                                                                               h_p.to(torch.double),
                                                                                               xOps.to(torch.double),
                                                                                               hOps.to(torch.double),
                                                                                               grad_h.to(torch.double))

        # grad_c_p and grad_h_p are zero because they are not trainable parameters
        return grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps

###########################################################################
###########################################################################


class Categorical_LSTM_Op_pass(Function):
    @staticmethod
    def forward(ctx, input, c_p, h_p, xOps, hOps):
        c, h, Xs = rnn_passes_cpp.Categorical_LSTM_Op_forward(input.to(torch.double),
                                                              c_p.to(torch.double),
                                                              h_p.to(torch.double),
                                                              xOps.to(torch.double),
                                                              hOps.to(torch.double))

        ctx.save_for_backward(input, Xs, c, c_p, h_p, xOps, hOps)
        ## when backprop is called, gradients passed to backward calls
        ## will correspond directly to the outputs of forward alls
        return c, h, Xs

    @staticmethod
    def backward(ctx, grad_c, grad_h, grad_Xs):
        ## grad_c and grad_Xs are zero because c and Xs from forward are not passed to the next layer
        input, Xs, c, c_p, h_p, xOps, hOps = ctx.saved_tensors
        grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps = rnn_passes_cpp.LSTM_Op_backward(input.to(torch.double),
                                                                                               Xs.to(torch.double),
                                                                                               c.to(torch.double),
                                                                                               c_p.to(torch.double),
                                                                                               h_p.to(torch.double),
                                                                                               xOps.to(torch.double),
                                                                                               hOps.to(torch.double),
                                                                                               grad_h.to(torch.double))

        # grad_c_p and grad_h_p are zero because they are not trainable parameters
        return grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps


###########################################################################
###########################################################################

class MC_LSTM_Op_pass(Function):
    @staticmethod
    def forward(ctx, input, c_p, h_p, xOps, hOps):
        c, h, Xs = rnn_passes_cpp.MC_LSTM_Op_forward(input.to(torch.double),
                                                     c_p.to(torch.double),
                                                     h_p.to(torch.double),
                                                     xOps.to(torch.double),
                                                     hOps.to(torch.double))
        ctx.save_for_backward(input, Xs, c, c_p, h_p, xOps, hOps)
        return c, h, Xs

    @staticmethod
    def backward(ctx, grad_c, grad_h, grad_Xs):
        ## grad_c and grad_Xs are zero because c and Xs from forward are not passed to the next layer
        input, Xs, c, c_p, h_p, xOps, hOps = ctx.saved_tensors
        
        grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps = rnn_passes_cpp.MC_LSTM_Op_backward(input.to(torch.double),
                                                                                                  Xs.to(torch.double),
                                                                                                  c.to(torch.double),
                                                                                                  c_p.to(torch.double),
                                                                                                  h_p.to(torch.double),
                                                                                                  xOps.to(torch.double),
                                                                                                  hOps.to(torch.double),
                                                                                                  grad_h.to(torch.double))
        # grad_c_p and grad_h_p are zero because they are not trainable parameters
        return grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps

    
# LSTM_Op is an LSTM layer whose kernels are true operators rather than Hadamard-like operators
# biases are not used
# multi-layer functionality is not currently supported
# bidirectionality is not currently supported

class LSTM_Op(nn.Module):
    def __init__(self, batch_size, input_channels, input_dim, output_channels, output_dim):
        super(LSTM_Op, self).__init__()
        self.batch_size = batch_size
        self.input_channels = input_channels        
        self.input_dim = input_dim
        
        self.output_channels = output_channels
        self.output_dim = output_dim
        
        self.state_size = batch_size*input_channels*output_channels*output_dim
        self.hOp_state_size = self.state_size*output_dim
        self.xOp_state_size = self.state_size*input_dim        
        
        self.c = torch.zeros([batch_size, output_channels, output_dim])
        self.h = torch.full([batch_size, output_channels, output_dim], 1)
        self.h /= self.batch_size*self.output_channels*self.output_dim

        # operators act to the right -> Op*input
        # x operators map input sequences of length input_dim to sequences of length output_dim
        # h operators are square matrices of dimension hidden_size
        xOps = torch.randn([4, self.output_channels, self.input_channels, self.output_dim, self.input_dim])
        hOps = torch.randn([4, self.output_channels, self.output_channels, self.output_dim, self.output_dim])

        xOps /= math.sqrt(self.xOp_state_size)
        hOps /= math.sqrt(self.hOp_state_size)
        
        self.xOps = nn.Parameter(xOps)
        self.hOps = nn.Parameter(hOps)        

    def reset_states(self):    
        self.c = torch.zeros([self.batch_size, self.output_channels, self.output_dim])
        self.h = torch.full([self.batch_size, self.output_channels, self.output_dim], 1)
        self.h /= self.batch_size*self.output_channels*self.output_dim

    def load_states(self, c, h):
        self.c = c
        self.h = h
        
        return

    def detach_states(self):
        self.c = self.c.detach()
        self.h = self.h.detach()

        return
    
    def forward(self, input):
        c, h, _ = LSTM_Op_pass.apply(input, self.c, self.h, self.xOps, self.hOps)

        self.c = c
        self.h = h

        return self.h

    
########################################################################################
########################################################################################
# an LSTM_Op that predicts whether the price will move up or down by a certain fraction

class Categorical_LSTM_Op(nn.Module):
    def __init__(self, batch_size, input_channels, input_dim):
        super(Categorical_LSTM_Op, self).__init__()
        self.batch_size = batch_size
        self.input_channels = input_channels        
        self.input_dim = input_dim
        
        
        self.hOp_state_size = batch_size*input_channels
        self.xOp_state_size = batch_size*input_channels
        
        self.c = torch.zeros([batch_size, 4])
        self.h = torch.randn([batch_size, 4])

        # operators act to the right -> Op*input
        # x operators map input sequences of length input_dim to sequences of length output_dim
        # h operators are square matrices of dimension hidden_size
        xOps = torch.randn([4, self.input_channels, self.input_dim, 4])
        hOps = torch.randn([4, 4, 4])

        xOps /= math.sqrt(self.xOp_state_size)
        hOps /= math.sqrt(self.hOp_state_size)
        
        self.xOps = nn.Parameter(xOps)
        self.hOps = nn.Parameter(hOps)        

    def reset_states(self):    
        self.c = torch.zeros([self.batch_size, 4])
        self.h = torch.randn([self.batch_size, 4])        

    def load_states(self, c, h):
        self.c = c
        self.h = h
        
        return

    def detach_states(self):
        self.c = self.c.detach()
        self.h = self.h.detach()

        return
    
    def forward(self, input):
        c, h, _ = Categorical_LSTM_Op_pass.apply(input, self.c, self.h, self.xOps, self.hOps)

        self.c = c
        self.h = h
        
        return self.h

    
    
########################################################################################
########################################################################################
# same as LSTM_Op but with support for multiple input currencies

class MC_LSTM_Op(nn.Module):
    def __init__(self, batch_size, input_currencies, input_channels, input_dim, output_currencies, output_channels, output_dim):
        super(MC_LSTM_Op, self).__init__()
        self.batch_size = batch_size
        self.input_currencies = input_currencies
        self.input_channels = input_channels        
        self.input_dim = input_dim

        self.output_currencies = output_currencies
        self.output_channels = output_channels
        self.output_dim = output_dim

        self.state_size = batch_size*input_currencies*input_channels*output_currencies*output_channels*output_dim
        self.xOp_state_size = self.state_size*input_dim
        self.hOp_state_size = self.state_size*output_dim
        
        self.hOp_state_size = self.state_size*input_dim
        self.xOp_state_size = self.state_size*output_dim        
        
        self.c = torch.zeros([batch_size, output_currencies, output_channels, output_dim])
        self.h = torch.full([batch_size, output_currencies, output_channels, output_dim], 1)
        self.h /= self.batch_size*self.output_currencies*self.output_channels*self.output_dim

        
        # operators act to the right -> Op*input
        # x operators map input sequences of length input_dim to sequences of length output_dim
        # h operators are square matrices of dimension hidden_size
        xOps = torch.randn([4, self.output_currencies, self.input_currencies, self.output_channels, self.input_channels, self.output_dim, self.input_dim])
        hOps = torch.randn([4, self.output_currencies, self.output_currencies, self.output_channels, self.output_channels, self.output_dim, self.output_dim])
        
        xOps /= math.sqrt(self.xOp_state_size)
        hOps /= math.sqrt(self.hOp_state_size)
        
        self.xOps = nn.Parameter(xOps)
        self.hOps = nn.Parameter(hOps)

    def reset_states(self):        
        self.c = torch.zerosl([self.batch_size, self.output_currencies, self.output_channels, self.output_dim])
        self.h = torch.full([self.batch_size, self.output_currencies, self.output_channels, self.output_dim], 1)
        self.h /= self.batch_Size*self.output_currencies*self.output_channels*self.output_dim

    def load_states(self, c, h):
        self.c = c
        self.h = h
        
        return

    def detach_states(self):
        self.c = self.c.detach()
        self.h = self.h.detach()

        return
        
    def forward(self, input):
        c, h, _ = MC_LSTM_Op_pass.apply(input, self.c, self.h, self.xOps, self.hOps)
        self.c, self.h = c, h
        
        return self.h

