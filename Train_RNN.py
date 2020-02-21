#########################################################################
##  BPI_RNN.py
##  Feb. 2020 - J. Hill
#########################################################################

from __future__ import absolute_import, division
import sys
import os

sys.path.append(os.path.join(os.getcwd(),'build_tools'))

import pathlib
import MODEL_CONSTANTS
import BPI_RNN
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt

from optparse              import OptionParser
from torch.autograd        import Function
from torch.utils.data      import Dataset, DataLoader
"""

"""

#########################################################################
##  preprocess data
#########################################################################

## import data file
cwd = os.getcwd()
data_path = os.path.join(cwd,'data')
data_path = os.path.join(data_path,'BTC_USD_2013-10-01_2020-01-28-CoinDesk.csv')

pd_object = pd.read_csv(data_path,usecols=['Date','24h High (USD)','24h Low (USD)'])
np_object = pd_object.to_numpy()

## collect dates and prices from np_object
dates          =             np_object[:,0]
prices_raw     =             np.asarray(np_object[:,1:], dtype=np.float64)

## express each element as the relative difference between neighboring prices
prices_div      =           np.asarray(prices_raw[:-1])
prices_num      =            np.asarray(prices_raw[1:])
prices           =            np.divide(prices_num, prices_div) - 1

## convert array to torch tensors
prices = torch.from_numpy(prices)


class BTC_Dataset(Dataset):
    def __init__(self, data, START, END, history_size, target_size, overlap):
        super(BTC_Dataset, self).__init__()
        self.history_size = history_size
        self.target_size = target_size

        self.ds = []

        if END is None: END = len(data)

        END -= target_size-overlap
        
        for i in range(START+history_size,END):
            x_idxs = range(i-history_size, i)
            x = data[x_idxs]

            y_idxs = range(i-overlap, i-overlap+target_size)
            y = data[y_idxs]
            
            self.ds.append([x,y])
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x, y = self.ds[idx][0], self.ds[idx][1]
        x, y = torch.tensor(x), torch.tensor(y)
        
        return {'x': x, 'y': y}
    
## auxiliary plotting function for visualization
def plot_data(arrays, delta, title, filename):
    fig = plt.figure(figsize=(10,10))
    labels = [['True high', 'True low'], ['Predicted high', 'Predicted low']]
    colors = [['blue','red'], ['green','yellow']]
    
    past = arrays[0].shape[0]
    future = list(range(delta))
    time_steps = range(-past,0) 
    plt.title(title)

    for i,x in enumerate(arrays):
        if i:
            ## array indices are (list index,time_step,high/low)
            plt.plot(future, arrays[i][:,0], color=colors[i][0], marker='.', markersize=1, label=labels[i][0])
            plt.plot(future, arrays[i][:,1], color=colors[i][1], marker='.', markersize=1, label=labels[i][1])

        else:
            plt.plot(future, arrays[i][:,0], color=colors[i][0], marker='.', markersize=1, label=labels[i][0])
            plt.plot(future, arrays[i][:,1], color=colors[i][1], marker='.', markersize=1, label=labels[i][1])

    plt.legend()
    plt.xlim(xmin=time_steps[0], xmax=(delta+5)*2)
    plt.xlabel('time step (d)')
    plt.savefig(filename+'.pdf')        
    plt.close(fig)
    


#########################################################################
##  create datasets and import training parameters
#########################################################################

train_frac  =                               0.6
LENGTH      =                       len(prices)
START_IDX   =                                 0
TRAIN_SPLIT =   int(np.ceil(train_frac*LENGTH))
HIST_SIZE   =         MODEL_CONSTANTS.HIST_SIZE
TARG_SIZE   =         MODEL_CONSTANTS.TARG_SIZE
HIDDEN_SIZE =       MODEL_CONSTANTS.HIDDEN_SIZE
OVERLAP     =       int(np.ceil(0.5*HIST_SIZE))

BATCH_SIZE  =        MODEL_CONSTANTS.BATCH_SIZE
BUFFER_SIZE =                              1000
NUM_WORKERS =                                 1
EPOCHS           =                                                   range(100)


train_ds = BTC_Dataset(prices, START_IDX, TRAIN_SPLIT, HIST_SIZE, TARG_SIZE, OVERLAP)
test_ds  = BTC_Dataset(prices, TRAIN_SPLIT, None, HIST_SIZE, TARG_SIZE, OVERLAP)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

train_size = len(train_dl)
test_size = len(train_dl)

#########################################################################
##  create/load and train the model
#########################################################################

LSTM_MODEL_SAVE_PATH = "./build_tools/lstm_saves.pth"

lstm_model = BPI_RNN.LSTM_Model(BATCH_SIZE, HIST_SIZE, TARG_SIZE, HIDDEN_SIZE)
lstm_model.load_state_dict(torch.load(LSTM_MODEL_SAVE_PATH))

optimizer = optim.Adam(lstm_model.parameters())

train_MSE_hist = []
test_MSE_hist  = []

status_labels = ['training loss', 'testing loss']

print('|{: ^10}'.format('epoch'), end='')
for lbl in status_labels:
    print('|{: ^30}'.format(lbl), end='')
print('|')

for epoch in EPOCHS:
    train_loss = 0.0
    test_loss = 0.0

    for data in train_dl:
        x, y = data['x'], data['y']

        optimizer.zero_grad()
        y_pred = lstm_model(x)
        loss = F.mse_loss(y_pred, y, reduction='mean')

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_MSE_hist.append(train_loss/train_size)

    with torch.no_grad():
        for data in test_dl:
            x, y = data['x'], data['y']

            y_pred = lstm_model(x)
            loss = F.mse_loss(y_pred, y, reduction='mean')

            test_loss += loss.item()

        test_MSE_hist.append(test_loss/test_size)

    nums = [train_MSE_hist[-1], test_MSE_hist[-1]]
    print('|{: >10}'.format(epoch), end='')
    for num in nums:
        print('|{: >30.3f}'.format(num), end='')
    print('|')

plt.figure(num=0, figsize=(10,10))
plt.plot(EPOCHS, np.asarray(train_MSE_hist), c='blue', linestyle='-', label='training loss')
plt.plot(EPOCHS, np.asarray(test_MSE_hist), c='red', linestyle='--', label='testing loss')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('MSE over epochs')
plt.legend()
plt.savefig('RNN_MSE.pdf')
plt.close(0)
