#########################################################################
##  Aux.py
##  March 2020 - J. Hill
#########################################################################

from __future__ import absolute_import, division
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'build_tools'))

import time
import random
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
import mpl_toolkits.mplot3d

from matplotlib.collections import PolyCollection
from torch.autograd         import Function
from torch.utils.data       import Dataset, DataLoader
from optparse               import OptionParser

"""
An auxiliary file containing all routines for preprocessing raw data into training and testing sets
Also included are visualization and postprocessing functions for evaluating the model's performance
"""

COINDESK_LABELS = ['Close', 'Open', 'High', 'Low']
YAHOO_LABELS = ['Open', 'High', 'Low', 'Close']
STANDARD_LABELS = ['High', 'Low', 'Open', 'Close']

NUM_INPUT_CHANNELS  = MODEL_CONSTANTS.NUM_INPUT_CHANNELS
NUM_OUTPUT_CHANNELS = MODEL_CONSTANTS.NUM_OUTPUT_CHANNELS

COINDESK_AXES_MAP = [[0,2],[1,3],[2,3]]
YAHOO_AXES_MAP = [[0,1],[1,2]]

if (MODEL_CONSTANTS.MC == 0):
    if (MODEL_CONSTANTS.CURRENCY == 'BTC' or MODEL_CONSTANTS.CURRENCY == 'ETH'): LABELS = COINDESK_LABELS
    elif (MODEL_CONSTANTS.CURRENCY == 'LTC'): LABELS = YAHOO_LABELS


#########################################################################
##  preprocess data
#########################################################################

## import data file
cwd = os.getcwd()
data_path = os.path.join(cwd,'data')

def prepare_data(currency, multi_currency, dictionary):
    print('-preparing data')
    pd_objects = []
    
    
    if (multi_currency == 1):

        for ENTRY in MODEL_CONSTANTS.LOADING_DICT:
            file_path = ENTRY['DATA_FILE']
            print('     -loading {}'.format(file_path))            
            pd_object = pd.read_csv(os.path.join(data_path,file_path), usecols=ENTRY['usecols'])

            for idx in ENTRY['axes_map']:
                pd_object = pd_object.swapaxes(pair[0], pair[1])
                
            pd_objects.append(pd_object)


        ## find shortest data array and truncate all others to the same length
        length = 1e9
        for pd_object in pd_objects:
            if (len(pd_object) < length): length = len(pd_object)

        dates = pd_objects[0]['Date'][-length:]

        print('-truncating all data arrays to length {}'.format(length))

        for i in range(len(pd_objects)):
            pd_objects[i] = pd_objects[i][-length:]
            pd_objects[i] = pd_objects[i].drop(labels='Date', axis=1)
            pd_objects[i] = pd_objects[i].to_numpy()
                        
        dims = [len(pd_objects), pd_objects[0].shape[0], pd_objects[0].shape[1]]
        np_object = np.empty(dims)

        ## populate np_object with reordered pd_objects
        for i in range(len(pd_objects)):
            j = 0
            
            for idx in ENTRY['axes_map']:
                np_object[i][:,j] = pd_objects[i][:,idx]
                j += 1


        ## shape of prices_raw is (currency, timestep, channel)
        prices_raw     =             np.asarray(np_object, dtype=np.float64)
        prices_raw     =             prices_raw[MODEL_CONSTANTS.START_IDX:]
        
        ## express each element as the relative difference between neighboring prices
        prices_div      =            np.asarray(prices_raw[:,:-1])
        prices_num      =            np.asarray(prices_raw[:,1:])
        prices          =            np.divide(prices_num, prices_div) - 1

        ## convert array to torch tensors and reshape to (currency, channel, timestep)
        prices_pt       =            torch.transpose(torch.from_numpy(prices), 1, 2)
        ref_prices_pt   =            torch.transpose(torch.from_numpy(prices_raw), 1, 2)
        
        return dates, prices_pt, ref_prices_pt

    elif (multi_currency == 0):
        data_file = None
        axes_map = None
        
        if   (MODEL_CONSTANTS.CURRENCY == 'BTC'): data_file = MODEL_CONSTANTS.BTC_DATA_FILE
        elif (MODEL_CONSTANTS.CURRENCY == 'ETH'): data_file = MODEL_CONSTANTS.ETH_DATA_FILE
        elif (MODEL_CONSTANTS.CURRENCY == 'LTC'): data_file = MODEL_CONSTANTS.LTC_DATA_FILE
        elif (MODEL_CONSTANTS.CURRENCY == 'MKR'): data_file = MODEL_CONSTANTS.MKR_DATA_FILE
        
        file_path = os.path.join(data_path, data_file)
        print('     -loading {}'.format(data_file))
        
        if (currency == 'BTC' or currency == 'ETH'):
            pd_object = pd.read_csv(file_path, usecols=
                                    ['Date','Closing Price (USD)','24h Open (USD)','24h High (USD)', '24h Low (USD)'])
            axes_map = [2,3,1,0]

        elif (currency == 'LTC' or currency == 'MKR'):
            pd_object = pd.read_csv(file_path, usecols=
                                    ['date','close','high','low','open'])
            axes_map = [1,2,3,0]

        ## isolate dates
        dates = pd_object['Date']
        pd_object = pd_object.drop(labels='Date', axis=1)
        pd_object = pd_object.to_numpy()
        
        np_object = np.empty_like(pd_object)
        j = 0
        
        for idx in axes_map:
            np_object[:,j] = pd_object[:,idx]
            j += 1

        ## shape of prices_raw is (timestep, channel)
        prices_raw     =             np.asarray(np_object, dtype=np.float64)

        ## express each element as the relative difference between neighboring prices
        prices_div      =            np.asarray(prices_raw[:-1])
        prices_num      =            np.asarray(prices_raw[1:])
        prices          =            np.divide(prices_num, prices_div) - 1

        ## convert array to torch tensors and reshape to (channel, timestep)
        prices_pt       =            torch.transpose(torch.from_numpy(prices), 0, 1)
        ref_prices_pt   =            torch.transpose(torch.from_numpy(prices_raw), 0, 1)

        return dates, prices_pt, ref_prices_pt


def integrate_output(ref_prices_pt, ref_idxs, output):
    #create high/low reference prices
    #ref_idxs corresponds to the first set of indices in the dataset
    #plug this directly into ref_prices_pt because it contains one earlier timestep
    ref_idxs = torch.squeeze(ref_idxs, 0)

    if   (MODEL_CONSTANTS.MC == 0):
        ref_prices = ref_prices_pt[:,ref_idxs[0]]
        y = torch.empty_like(output)
        timesteps = output.shape[-1]

        for i in range(timesteps):
            prices = ref_prices * (output[:,i]+1)
            ref_prices = prices
            y[:,i] = prices

    elif (MODEL_CONSTANTS.MC == 1):
        ref_prices = ref_prices_pt[:,:,ref_idxs[0]]
        y = torch.empty_like(output)
        timesteps = output.shape[-1]

        for i in range(timesteps):
            prices = ref_prices * (output[:,:,i]+1)
            ref_prices = prices
            y[:,:,i] = prices
        
        
    return y

def percent_error(pred_vals, true_vals):
    f = 100*(pred_vals-true_vals)/true_vals
    return f


#########################################################################333
## cryptocurrency dataset for training and testing the network
class Crypto_Dataset(Dataset):
    def __init__(self, data, START, END, history_size, target_size, overlap):
        super(Crypto_Dataset, self).__init__()
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
        
        idx += self.start
        x_idxs = range(idx-self.history_size+1, idx+1)
        x = self.data[:NUM_INPUT_CHANNELS,x_idxs]
        
        y_idxs = range(idx-self.overlap+1, idx-self.overlap+self.target_size+1)
        y_idxs = torch.tensor(y_idxs)
        y = self.data[:NUM_OUTPUT_CHANNELS,y_idxs]

        return {'x': x, 'y': y, 'y_idxs': y_idxs}

    def get_item_alt(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        idx += self.start
        x_idxs = range(idx-self.history_size+1, idx+1)
        x = self.data[:NUM_INPUT_CHANNELS,x_idxs]

        y_idxs = range(idx-self.overlap+1, idx-self.overlap+self.target_size+1)
        y_idxs = torch.tensor(y_idxs)
        y = None

        return {'x': x, 'y': y, 'y_idxs': y_idxs}

## special first-order correction dataset for gradient-boosted models
class NLO_Crypto_Dataset(Dataset):
    ## this class is initialized with a Crypto_Dataset class as its data and a trained model
    def __init__(self, LO_dataset, NLO_dataset):
        super(NLO_Crypto_Dataset, self).__init__()
        self.dataset = LO_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        data = self.dataset[idx]
        x = data['x']
        delta_y = data['y']
        y_idxs = data['y_idxs']
        
        return {'x': x, 'y': delta_y, 'y_idxs': y_idxs}

    def get_item_alt(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        return self.dataset.get_item_alt(idx)

## create a next-to-leading-order dataset
def create_NLO_Crypto_Dataset(dataset, models):
    LO_dataset = dataset
    NLO_dataset = []

    with torch.no_grad():
        for data in dataset:
            x, y, y_idxs = data['x'], data['y'], data['y_idxs']
            x, y, y_idxs = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0), torch.unsqueeze(y_idxs, 0)

            y_pred = torch.zeros_like(y)
            
            for model in models:
                y_pred += model(x)
                
            delta_y = y - y_pred

            x, delta_y, y_idxs = torch.squeeze(x, 0), torch.squeeze(delta_y, 0), torch.squeeze(y_idxs, 0)
            NLO_dataset.append({'x': x, 'y': delta_y, 'y_idxs': y_idxs})

    return NLO_Crypto_Dataset(LO_dataset, NLO_dataset)

#########################################################################333
## multi-currency dataset for training and testing the network
class MC_Crypto_Dataset(Dataset):
    def __init__(self, data, START, END, history_size, target_size, overlap):
        super(MC_Crypto_Dataset, self).__init__()
        self.history_size = history_size
        self.target_size = target_size

        self.start = START + history_size
        self.overlap = overlap
        self.data = data

        if END is None: END = len(self.data[0,0])
        END -= target_size-overlap

        self.end = END
        
    def __len__(self):
        return self.end-self.start

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        
        idx += self.start
        x_idxs = range(idx-self.history_size+1, idx+1)
        x = self.data[:,:NUM_INPUT_CHANNELS,x_idxs]
        
        y_idxs = range(idx-self.overlap+1, idx-self.overlap+self.target_size+1)
        y_idxs = torch.tensor(y_idxs)
        y = self.data[:,:NUM_OUTPUT_CHANNELS,y_idxs]

        return {'x': x, 'y': y, 'y_idxs': y_idxs}

    def get_item_alt(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        idx += self.start
        x_idxs = range(idx-self.history_size+1, idx+1)
        x = self.data[:,:NUM_INPUT_CHANNELS,x_idxs]

        y_idxs = range(idx-self.overlap+1, idx-self.overlap+self.target_size+1)
        y_idxs = torch.tensor(y_idxs)
        y = None

        return {'x': x, 'y': y, 'y_idxs': y_idxs}


## auxiliary plotting function for visualization
def plot_data(true_values, predictions, true_idxs, pred_idxs, hist_size, targ_size, overlap, dates, title, labels, filename):
    fig = plt.figure(figsize=(8,6))

    if (MODEL_CONSTANTS.VER == 1):
        labels = [['True high', 'True low'], ['Predicted high', 'Predicted low']]
        colors = ['blue','red']

    elif (MODEL_CONSTANTS.VER == 2):
        colors = ['blue','red','orange','green']
        
    # create tick marks for every day and labels for every 10th day    
    x_ticks = np.arange(true_idxs[0], pred_idxs[-1]+5)
    tick_labels = []
    ticks = []

    delta_T = x_ticks[-1]-x_ticks[0]+1    
    new_dates = pd.date_range(dates[x_ticks[0]+1], periods=delta_T)

    for i in range(delta_T):
        if (i % 2 == 0):
            tick_labels.append(new_dates[i].date())
            ticks.append(x_ticks[i])
        else: tick_labels.append('')
    
    plt.title(title)

    ## plot true values
    for i in range(NUM_INPUT_CHANNELS):
        plt.plot(true_idxs, true_values[i][true_idxs], color=colors[i], marker='.', markersize=1, label=labels[i])

    ## plot predicted values
    for i in range(NUM_OUTPUT_CHANNELS):
        plt.plot(pred_idxs, predictions[i], color=colors[i], linestyle='--', marker='.', markersize=1)
    
    plt.legend()
    plt.xlim(xmin=true_idxs[0], xmax=(pred_idxs[-1]+5))
    plt.xticks(x_ticks, labels=tick_labels)

    #y_ticks = plt.get_yticks();
    #step = (y_ticks[1]-y_ticks[0])/10
    #y_minor_ticks = np.arange(y_ticks[0], y_ticks[-1], step=step)
    #plt.set_yticks(y_minor_ticks, minor=True)
    
    plt.xlabel('date')
    plt.ylabel('value ($)')
    
    plt.grid(axis='x')
    plt.savefig(filename+'.pdf')        
    plt.close(fig)

    
# construct the vertex list that defines the polygon representing each curve
# borrowed from matplotlib docs
def polygon_under_graph(x_list, y_list, ymin):
    return [(x_list[0], ymin), *zip(x_list, y_list), (x_list[-1], ymin)]
    
def plot_prediction_surface(true_values, predictions, idxs, targ_size, title, filename):
    # input array indices are like [dataset idx][high/low][timestep]

    y_lo_min, y_hi_min = None, None

    for prediction in predictions:
        imin = np.argmin(prediction[0])        
        if (y_lo_min is None or prediction[0,imin] < y_lo_min): y_lo_min = prediction[0,imin]

        imin = np.argmin(prediction[1])
        if (y_hi_min is None or prediction[1,imin] < y_hi_min): y_hi_min = prediction[1,imin]
        
    labels = ['Predicted high', 'Predicted low']
    colors = ['red', 'orange', 'yellow', 'green']
    
    # only one set of x coords necessary because high/low timesteps are identical
    min_vertices, max_vertices = [], []

    i0, i = len(idxs), 0
    Z = range(-len(idxs), 0)
    
    for idx, prediction in zip(idxs, predictions):
        X = idx[-i0:targ_size-i].numpy()
        
        Y_low = prediction[1][-i0:targ_size-i].numpy()
        Y_hi = prediction[0][-i0:targ_size-i].numpy()

        min_vertices.append(polygon_under_graph(X, Y_low, y_lo_min))
        max_vertices.append(polygon_under_graph(X, Y_hi,  y_hi_min))

        i += 1
        
    min_polys = PolyCollection(min_vertices, facecolors=colors, alpha=0.6)
    max_polys = PolyCollection(max_vertices, facecolors=colors, alpha=0.6)

    # create 2D true value curve for projection on  yz plane
    true_x = idxs[0].numpy()[-i0:targ_size]
    true_y_lo = true_values[1].numpy()
    true_y_hi = true_values[0].numpy()
    
    fig = plt.figure(figsize=(10,8))    
    ax = fig.gca(projection='3d')

    ax.set_xlabel('timestep (d)')
    ax.set_ylabel('null label')
    ax.set_zlabel('value ($)')

    ax.set_xlim(idxs[0].numpy()[-i0]-1, idxs[0].numpy()[-1]+1)
    ax.set_ylim(-i0-1,1)
    ax.set_zlim(y_lo_min, 9e3)
    
    ax.add_collection3d(min_polys, Z, zdir='y')
    ax.plot(true_x, true_y_lo, zs=0, zdir='y')

    plt.title(title+'_PredictedHighs')
    plt.savefig(filename+'_PredictedHighs.pdf')    
    plt.close(fig)

    fig = plt.figure(figsize=(10,8))
    ax = fig.gca(projection='3d')
    
    ax.set_xlabel('timestep (d)')
    ax.set_ylabel('null label')
    ax.set_zlabel('value ($)')

    ax.set_xlim(idxs[0].numpy()[-i0]-1, idxs[0].numpy()[-1]+1)    
    ax.set_ylim(-i0-1,1)
    ax.set_zlim(y_hi_min, 9e3)

    ax.add_collection3d(max_polys, Z, zdir='y')
    ax.plot(true_x, true_y_hi, zs=0, zdir='y')    
    
    plt.title(title+'_PredictedLows')
    plt.savefig(filename+'_PredictedLows.pdf')    
    plt.close(fig)


## create dataset and dataloader
def create_ds_and_dl():
    print('-creating dataset and dataloader')
    if (MODEL_CONSTANTS.MC == 0):
        train_ds = Crypto_Dataset(prices_pt, MODEL_CONSTANTS.START_IDX, None, MODEL_CONSTANTS.HIST_SIZE,
                                  MODEL_CONSTANTS.TARG_SIZE, MODEL_CONSTANTS.OVERLAP)
        
        if (MODEL_CONSTANTS.ORDER > 0):
            LO_models = []
            LO_lstm_model = BPI_RNN.LSTM_Model(MODEL_CONSTANTS.BATCH_SIZE, MODEL_CONSTANTS.NUM_DATA_CHANNELS,
                                               MODEL_CONSTANTS.HIST_SIZE, MODEL_CONSTANTS.NUM_DATA_CHANNELS, MODEL_CONSTANTS.TARG_SIZE)
            
            for i in range(MODEL_CONSTANTS.ORDER):
                save_path, _, _ = MODEL_CONSTANTS.create_save_paths(MODEL_CONSTANTS.CURRENCY, MODEL_CONSTANTS.ORDER-(i+1), 1,
                                                                    MODEL_CONSTANTS.BATCH_SIZE, MODEL_CONSTANTS.HIST_SIZE,
                                                                    MODEL_CONSTANTS.TARG_SIZE, MODEL_CONSTANTS.OVERLAP, MODEL_CONSTANTS.LR)
    
                LO_lstm_model.load_state_dict(torch.load(save_path))
                LO_models.append(LO_lstm_model)
    
            train_ds = create_NLO_Crypto_Dataset(train_ds, LO_models)
            
    elif(MODEL_CONSTANTS.MC == 1):
        train_ds = MC_Crypto_Dataset(prices_pt, MODEL_CONSTANTS.START_IDX, None, MODEL_CONSTANTS.HIST_SIZE,
                                     MODEL_CONSTANTS.TARG_SIZE, MODEL_CONSTANTS.OVERLAP)
        
    train_dl = DataLoader(train_ds, batch_size=MODEL_CONSTANTS.BATCH_SIZE, shuffle=False, num_workers=1)        

    return train_ds, train_dl



#########################################################################
## create the essential arrays
#########################################################################
dates, prices_pt, ref_prices_pt = prepare_data(MODEL_CONSTANTS.CURRENCY, MODEL_CONSTANTS.MC, MODEL_CONSTANTS.LOADING_DICT)

