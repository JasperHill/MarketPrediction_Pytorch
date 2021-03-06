#########################################################################
##  Aux.py
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
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt

from torch.autograd        import Function
from torch.utils.data      import Dataset, DataLoader
from optparse              import OptionParser

"""
An auxiliary file containing all routines for preprocessing raw data into training and testing sets
Also included are visualization and postprocessing functions for evaluating the model's performance
"""

#########################################################################
##  preprocess data
#########################################################################

## import data file
cwd = os.getcwd()
data_path = os.path.join(cwd,'data')
data_path = os.path.join(data_path, MODEL_CONSTANTS.DATA_FILE)

pd_object = pd.read_csv(data_path,usecols=['Date','24h High (USD)','24h Low (USD)'])
np_object = pd_object.to_numpy()

## collect dates and prices from np_object
dates          =             np_object[:,0]
prices_raw     =             np.asarray(np_object[:,1:], dtype=np.float64)

## express each element as the relative difference between neighboring prices
prices_div      =            np.asarray(prices_raw[:-1])
prices_num      =            np.asarray(prices_raw[1:])
prices          =            np.divide(prices_num, prices_div) - 1

## convert array to torch tensors and reshape to (high/low, timestep)
prices_pt       =            torch.transpose(torch.from_numpy(prices), 0, 1)
ref_prices_pt   =            torch.transpose(torch.from_numpy(prices_raw), 0, 1)


def integrate_output(ref_prices_pt, ref_idxs, output):
    #create high/low reference prices
    ref_prices = ref_prices_pt[:,ref_idxs[0]]
    
    y = torch.empty_like(output)
    timesteps = output.shape[-1]

    for i in range(timesteps):
        prices = ref_prices * (output[:,i]+1)
        ref_prices = prices
        y[:,i] = prices

    return y

def percent_error(pred_vals, true_vals):
    f = 100*(pred_vals-true_vals)/true_vals
    return f

class BTC_Dataset(Dataset):
    def __init__(self, data, START, END, history_size, target_size, overlap):
        super(BTC_Dataset, self).__init__()
        self.history_size = history_size
        self.target_size = target_size

        self.start = START + history_size
        self.overlap = overlap
        self.data = data

        if END is None: END = len(self.data[0])
        END -= target_size-overlap

        self.end = END
        
    def __len__(self):
        return self.end-self.start

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        
        idx += self.history_size
        x_idxs = range(idx-self.history_size, idx)
        x = self.data[:,x_idxs]
        
        y_idxs = range(idx-self.overlap, idx-self.overlap+self.target_size)
        y_idxs = torch.tensor(y_idxs)
        y = self.data[:,y_idxs]

        return {'x': x, 'y': y, 'y_idxs': y_idxs}
    
## auxiliary plotting function for visualization
def plot_data(true_values, predictions, idxs, hist_size, targ_size, overlap, title, filename):
    fig = plt.figure(figsize=(10,8))
    labels = [['True high', 'True low'], ['Predicted high', 'Predicted low']]
    colors = [['blue','red'], ['cyan','magenta']]

    ref_range = range(idxs[0]-hist_size, idxs[-1]+1)
    plt.title(title)

    ## plot true values
    plt.plot(ref_range, true_values[0][ref_range], color=colors[0][0], marker='.', markersize=1, label=labels[0][0])
    plt.plot(ref_range, true_values[1][ref_range], color=colors[0][1], marker='.', markersize=1, label=labels[0][1])

    ## plot predicted values
    plt.plot(idxs, predictions[0], color=colors[1][0], marker='.', markersize=1, label=labels[1][0])
    plt.plot(idxs, predictions[1], color=colors[1][1], marker='.', markersize=1, label=labels[1][1])

    
    plt.legend()
    plt.xlim(xmin=idxs[0]-hist_size, xmax=(idxs[-1]+5))
    plt.xlabel('absolute time step (d)')
    plt.ylabel('value ($)')
    plt.savefig(filename+'.pdf')        
    plt.close(fig)
    


#########################################################################
##  create datasets and dataloaders
#########################################################################

train_frac  =        MODEL_CONSTANTS.TRAIN_FRAC
LENGTH      =                 len(prices_pt[0])
START_IDX   =                                 0
TRAIN_SPLIT =   int(np.ceil(train_frac*LENGTH))
HIST_SIZE   =         MODEL_CONSTANTS.HIST_SIZE
TARG_SIZE   =         MODEL_CONSTANTS.TARG_SIZE
OVERLAP     =           MODEL_CONSTANTS.OVERLAP

BATCH_SIZE  =        MODEL_CONSTANTS.BATCH_SIZE
BUFFER_SIZE =                              1000
NUM_WORKERS =                                 1


train_ds = BTC_Dataset(prices_pt, START_IDX, TRAIN_SPLIT, HIST_SIZE, TARG_SIZE, OVERLAP)
test_ds  = BTC_Dataset(prices_pt, TRAIN_SPLIT, None, HIST_SIZE, TARG_SIZE, OVERLAP)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

