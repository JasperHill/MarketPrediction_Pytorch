from __future__ import absolute_import, division

import os
import string
import torch
import MODEL_CONSTANTS
import torch.nn            as nn
import numpy               as np
import torch.nn.functional as F

from Custom_RNNs import LSTM_Op, MC_LSTM_Op

class LSTM_Model(nn.Module):
    def __init__(self, batch_size, input_channels, input_dim, output_channels, output_dim):
        super(LSTM_Model, self).__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim
        
        self.output_dim = output_dim
        self.output_channels = output_channels
        self.batch_size = batch_size

        self.lstm_op = LSTM_Op(batch_size, input_channels, input_dim, output_channels, output_dim)
        self.tanh    = nn.Tanh()

    def forward(self, x):
        h = self.lstm_op(x)
        
        return h

    def detach_states(self):
        self.lstm_op.detach_states()
        return

# TODO: create multi-channel LSTM_Model class capable of consisting of two layers of parallel LSTM_Ops
# that convolve data from multiple inputs to generate the same number of output tensors

class MC_LSTM_Model(nn.Module):
    def __init__(self, batch_size, input_currencies, input_channels, input_dim, output_currencies, output_channels, output_dim):
        super(MC_LSTM_Model, self).__init__()
        self.input_channels  = input_channels
        self.input_dim       = input_dim
        
        self.output_dim      = output_dim
        self.output_channels = output_channels
        self.batch_size      = batch_size

        self.mc_lstm_op  = MC_LSTM_Op(batch_size, input_currencies, input_channels, input_dim, output_currencies, output_channels, output_dim)
        self.tanh        = nn.Tanh()

    def forward(self, x):
        h = self.mc_lstm_op(x)

        return self.tanh(h)

    def detach_states(self):
        self.mc_lstm_op.detach_states()
        return

def save_model(Model):
    torch.save(Model.state_dict(), MODEL_CONSTANTS.MODEL_SAVE_PATH)
    
