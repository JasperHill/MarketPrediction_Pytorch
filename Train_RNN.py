#########################################################################
##  Train_RNN.py
##  Feb. 2020 - J. Hill
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

"""

"""
parser = OptionParser()
parser.add_option("-c", "--create_new_rnn", action="store_true", default=False, dest="create")
(options, args) = parser.parse_args()
create = options.create

#########################################################################
##  create datasets and import training parameters
#########################################################################

NUM_CURRENCIES      =      MODEL_CONSTANTS.NUM_CURRENCIES
NUM_INPUT_CHANNELS  =  MODEL_CONSTANTS.NUM_INPUT_CHANNELS
NUM_OUTPUT_CHANNELS = MODEL_CONSTANTS.NUM_OUTPUT_CHANNELS
TRAIN_FRAC          =          MODEL_CONSTANTS.TRAIN_FRAC
HIST_SIZE           =           MODEL_CONSTANTS.HIST_SIZE
TARG_SIZE           =           MODEL_CONSTANTS.TARG_SIZE
OVERLAP             =             MODEL_CONSTANTS.OVERLAP
THRESH              =              MODEL_CONSTANTS.THRESH

BATCH_SIZE          =          MODEL_CONSTANTS.BATCH_SIZE
EPOCHS              =   range(MODEL_CONSTANTS.NUM_EPOCHS)

train_ds, train_dl = Aux.create_ds_and_dl()

#########################################################################
##  create/load and train the model
#########################################################################

if (MODEL_CONSTANTS.CATEGORICAL == 0):
    if (MODEL_CONSTANTS.MC == 0):   lstm_model = BPI_RNN.LSTM_Model(BATCH_SIZE, NUM_INPUT_CHANNELS, HIST_SIZE, NUM_OUTPUT_CHANNELS, TARG_SIZE)
    elif (MODEL_CONSTANTS.MC == 1): lstm_model = BPI_RNN.MC_LSTM_Model(BATCH_SIZE, NUM_CURRENCIES, NUM_INPUT_CHANNELS, HIST_SIZE,
                                                                   NUM_CURRENCIES, NUM_OUTPUT_CHANNELS, TARG_SIZE)

    status_labels = ['MSE loss', 'epoch CPU time (min)']
    
elif (MODEL_CONSTANTS.CATEGORICAL == 1):
    lstm_model = BPI_RNN.Categorical_LSTM_Model(BATCH_SIZE, NUM_INPUT_CHANNELS, HIST_SIZE)
    status_labels = ['loss', 'epoch CPU time (min)']    

if create:
    BPI_RNN.save_model(lstm_model)

else:
    lstm_model.load_state_dict(torch.load(MODEL_CONSTANTS.MODEL_SAVE_PATH))

optimizer = optim.Adam(lstm_model.parameters(), lr=MODEL_CONSTANTS.LR)

train_MSE_hist = []
train_size = len(train_dl)





print('|{: ^10}'.format('epoch'), end='')
for lbl in status_labels:
    print('|{: ^30}'.format(lbl), end='')
print('|')

idx_str_length = 4
train_length_str = len(str(train_size))
full_start = time.time()

for epoch in EPOCHS:
    lstm_model.lstm_op.reset_states()
    epoch_start = time.time()

    train_loss = 0.0
    i = 0

    
    for data in train_dl:
        index_start = time.time()
        x, y = data['x'], data['y']
        optimizer.zero_grad()

        y_pred = lstm_model(x).to(torch.double)
        y = y.to(torch.double)
        
        loss = F.mse_loss(y_pred, y, reduction='mean')
        loss.backward(retain_graph=False)
        optimizer.step()

        ## important to detach hidden and cell states from autograd graph
        ## so that backpropagation does not proceed through all timesteps
        lstm_model.detach_states()
        
        train_loss += loss.item()
        index_end = time.time()
        index_time = index_end-index_start
        index_time /= 60
        
        idx_length = len(str(i))
        sys.stdout.write("\b" * 74)
        sys.stdout.write("|")
        sys.stdout.write(" "*(idx_str_length-idx_length))
        sys.stdout.write(str(i))
        sys.stdout.write("/")
        sys.stdout.write(str(train_size))
        sys.stdout.write(" "*(idx_str_length-train_length_str))        
        sys.stdout.write(" |")
        sys.stdout.write(" index time: ")
        sys.stdout.write('{: <17.1e}'.format(index_time))
        sys.stdout.write("|")
        sys.stdout.write(" total time: ")
        sys.stdout.write('{: <17.1e}'.format((time.time()-full_start)/60))
        sys.stdout.write("|")
        sys.stdout.flush()               

        i += 1
        
    train_MSE_hist.append(train_loss/train_size)

    
    epoch_end = time.time()
    duration = epoch_end-epoch_start
    
    nums = [train_loss/train_size, duration/60]
    sys.stdout.write("\b" * 74)
    sys.stdout.flush()
    print('|{: >10}'.format(epoch), end='')
    for num in nums:
        print('|{: >30.3e}'.format(num), end='')
    print('|')


print('-saving trained model')
torch.save(lstm_model.state_dict(), MODEL_CONSTANTS.MODEL_SAVE_PATH)
torch.save(lstm_model.lstm_op.c, MODEL_CONSTANTS.CELL_SAVE_PATH)
torch.save(lstm_model.lstm_op.h, MODEL_CONSTANTS.HIDDEN_SAVE_PATH)

# pass ref_prices_pt[:,1:] because training data is the relative change between timesteps
#Aux.plot_data(ref_prices_pt[:,1:], final_y, ref_idxs, HIST_SIZE, TARG_SIZE, OVERLAP, 'Bitcoin Price Over Time', 'Network_Prediction')

#pe = Aux.percent_error(final_y, ref_prices_pt[:,ref_idxs])

## plot network MSE over training epochs
plt.figure(num=0, figsize=(10,8))
plt.plot(np.asarray(train_MSE_hist), c='blue', linestyle='-')
plt.xlabel('index')
plt.ylabel('time (min)')
plt.title('MSE over epochs')
plt.legend()
plt.savefig('RNN_MSE.pdf')
plt.close(0)


