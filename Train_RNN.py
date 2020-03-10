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

HIST_SIZE   =         MODEL_CONSTANTS.HIST_SIZE
TARG_SIZE   =         MODEL_CONSTANTS.TARG_SIZE
OVERLAP     =           MODEL_CONSTANTS.OVERLAP

BATCH_SIZE  =        MODEL_CONSTANTS.BATCH_SIZE
EPOCHS      = range(MODEL_CONSTANTS.NUM_EPOCHS)

train_dl, test_dl = Aux.train_dl, Aux.test_dl

train_size = len(train_dl)
test_size = len(test_dl)

#########################################################################
##  create/load and train the model
#########################################################################


lstm_model = BPI_RNN.LSTM_Model(BATCH_SIZE, 2, HIST_SIZE, 2, TARG_SIZE)

if create:
    BPI_RNN.save_model(lstm_model)

else:
    lstm_model.load_state_dict(torch.load(MODEL_CONSTANTS.MODEL_SAVE_PATH))

optimizer = optim.Adam(lstm_model.parameters(), lr=MODEL_CONSTANTS.LR)

## reference prices for assessing network performance
ref_prices_pt = Aux.ref_prices_pt
final_ys, ref_idxs = None, None
train_MSE_hist = []
test_MSE_hist  = []

status_labels = ['training loss', 'testing loss', 'epoch CPU time (min)']

print('|{: ^10}'.format('epoch'), end='')
for lbl in status_labels:
    print('|{: ^30}'.format(lbl), end='')
print('|')

for epoch in EPOCHS:
    start = time.time()
    
    train_loss = 0.0
    test_loss = 0.0

    for data in train_dl:
        x, y = data['x'], data['y']

        optimizer.zero_grad()
        y_pred = lstm_model(x).to(torch.double)
        loss = F.mse_loss(y_pred, y, reduction='mean')

        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.item()

    train_MSE_hist.append(train_loss/train_size)

    with torch.no_grad():
        i = 0
        
        if (epoch == EPOCHS[-1]):
            j = random.randrange(0,len(test_dl))
        
        for data in test_dl:
            x, y, y_idxs = data['x'], data['y'], data['y_idxs']

            y_pred = lstm_model(x)
            loss = F.mse_loss(y_pred, y, reduction='mean')
            test_loss += loss.item()
            
            if ((epoch == EPOCHS[-1]) and (i == j)):
                final_pred = y_pred[-1]
                ref_idxs = y_idxs[-1]
            i += 1

        test_MSE_hist.append(test_loss/test_size)

    end = time.time()
    duration = end-start
    
    nums = [train_MSE_hist[-1], test_MSE_hist[-1], duration/(60*(epoch+1))]
    print('|{: >10}'.format(epoch), end='')
    for num in nums:
        print('|{: >30.3e}'.format(num), end='')
    print('|')

final_y = Aux.integrate_output(ref_prices_pt, ref_idxs, final_pred)

# pass ref_prices_pt[:,1:] because training data is the relative change between timesteps
Aux.plot_data(ref_prices_pt[:,1:], final_y, ref_idxs, HIST_SIZE, TARG_SIZE, OVERLAP, 'Bitcoin Price Over Time', 'Network_Prediction')

pe = Aux.percent_error(final_y, ref_prices_pt[:,ref_idxs])

## plot network MSE over training epochs
plt.figure(num=0, figsize=(10,8))
plt.plot(EPOCHS, np.asarray(train_MSE_hist), c='blue', linestyle='-', label='training loss')
plt.plot(EPOCHS, np.asarray(test_MSE_hist), c='red', linestyle='--', label='testing loss')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('MSE over epochs')
plt.legend()
plt.savefig('RNN_MSE.pdf')
plt.close(0)

## plot percent error of predictions
plt.figure(num=1, figsize=(10,8))
plt.plot(ref_idxs, pe[0], c='black', linestyle='-', label='high price')
plt.plot(ref_idxs, pe[1], c='grey', linestyle='-', label='low price')
plt.xlabel('absolute timestep (d)')
plt.ylabel('error (%)')
plt.title('Network Performance')
plt.legend()
plt.savefig('RNN_PE.pdf')
plt.close(1)


torch.save(lstm_model.state_dict(), MODEL_CONSTANTS.MODEL_SAVE_PATH)
