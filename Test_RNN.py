#########################################################################
##  Test_RNN.py
##  March 2020 - J. Hill
#########################################################################

from __future__ import absolute_import, division
import sys
import os

sys.path.append(os.path.join(os.getcwd(),'build_tools'))

import time
import random
import pathlib
import MODEL_CONSTANTS
import BPI_RNN
import Aux
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt

from torch.autograd        import Function
from optparse              import OptionParser


HIST_SIZE   =         MODEL_CONSTANTS.HIST_SIZE
TARG_SIZE   =         MODEL_CONSTANTS.TARG_SIZE
OVERLAP     =           MODEL_CONSTANTS.OVERLAP

BATCH_SIZE  =        MODEL_CONSTANTS.BATCH_SIZE

test_ds = Aux.test_ds

parser = OptionParser()
parser.add_option("-i", "--index", action="store", dest="idx",
                  help="specify a dataset index to propagate through the RNN (must be less than "+len(test_ds)")")
(options, args) = parser.parse_args()
idx = int(options.idx)

"""

"""

if (idx >= len(test_ds)): print('Error: specified index must be less than {}'.format(len(test_ds)-1))

#########################################################################
##  load the model and propagate a test sample
#########################################################################


lstm_model = BPI_RNN.LSTM_Model(BATCH_SIZE, 2, HIST_SIZE, 2, TARG_SIZE)
lstm_model.load_state_dict(torch.load(MODEL_CONSTANTS.MODEL_SAVE_PATH))

## reference prices for assessing network performance
ref_prices_pt = Aux.ref_prices_pt
final_ys, ref_idxs = None, None

with torch.no_grad():
    data = test_ds[idx]    
    x, y, y_idxs = data['x'], data['y'], data['y_idxs']
    x, y, y_idxs = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0), torch.unsqueeze(y_idxs,0)

    y_pred = lstm_model(x)
    final_pred = y_pred[-1]
    ref_idxs = y_idxs[-1]

print(y_pred.shape)
final_y = Aux.integrate_output(ref_prices_pt, ref_idxs, final_pred)

# pass ref_prices_pt[:,1:] because training data is the relative change between timesteps
Aux.plot_data(ref_prices_pt[:,1:], final_y, ref_idxs, HIST_SIZE, TARG_SIZE, OVERLAP, 'Bitcoin Price Over Time', 'Network_Test')
pe = Aux.percent_error(final_y, ref_prices_pt[:,ref_idxs])
