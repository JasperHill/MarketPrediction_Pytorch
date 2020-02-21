from __future__ import absolute_import, division

import os
import string
import torch
import MODEL_CONSTANTS
import torch.nn            as nn
import numpy               as np
import torch.nn.functional as F

from Custom_RNNs import LSTM_Op
from optparse    import OptionParser

parser = OptionParser()
parser.add_option("--save_model", action='store_true', dest="save")
options, args = parser.parse_args()
save = options.save

##################################################################################
##  set model hyperparameters
##################################################################################

HIST_SIZE        = MODEL_CONSTANTS.HIST_SIZE
TARG_SIZE        = MODEL_CONSTANTS.TARG_SIZE
HIDDEN_SIZE      = MODEL_CONSTANTS.HIDDEN_SIZE
BATCH_SIZE       = MODEL_CONSTANTS.BATCH_SIZE

class LSTM_Model(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, hidden_size):
        super(LSTM_Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.lstm_op = LSTM_Op(batch_size, input_dim, output_dim, hidden_size)
        self.linear  = nn.Linear(in_features=(output_dim+hidden_size), out_features=output_dim)
        self.tanh    = nn.Tanh()

    def forward(self, x):
        x = self.lstm_op(x)
        x = self.linear(x)

        return self.tanh(x)

LSTM_MODEL_SAVE_PATH = "./lstm_saves.pth"

lstm_model = LSTM_Model(BATCH_SIZE, HIST_SIZE, TARG_SIZE, HIDDEN_SIZE)

if save:
    torch.save(lstm_model.state_dict(), LSTM_MODEL_SAVE_PATH)
