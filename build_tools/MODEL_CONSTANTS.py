import os
import sys
import DATA_CONSTANTS

## MC toggles the multi-currency functionality
MC          =    0

CURRENCY    = 'BTC'
ORDER       =    0
START_IDX   =    0
TRAIN_FRAC  =    1
BATCH_SIZE  =    1
HIST_SIZE   =  400
TARG_SIZE   =    5
OVERLAP     =    0
NUM_EPOCHS  =    5
LR          =   1e-3

## variable determines the codebase version
## 1 uses only high/low prices as training data
## 2 uses high/low + open/close as training data

#
## VER = 1 is now deprecated and will not be compatible with future changes to the codebase
#

VER        =    2

NUM_INPUT_CHANNELS  = 4
NUM_OUTPUT_CHANNELS = 2
NUM_DATA_CHANNELS   = 4

BTC_DATA_FILE = DATA_CONSTANTS.BTC_DATA_FILE
ETH_DATA_FILE = DATA_CONSTANTS.ETH_DATA_FILE
LTC_DATA_FILE = DATA_CONSTANTS.LTC_DATA_FILE
MKR_DATA_FILE = DATA_CONSTANTS.MKR_DATA_FILE

# the axes_map entry is used in standardizing the columns of the numpy arrays
# so that the order goes high, low, open, close
LOADING_DICT = [{'currency': 'BTC',
                'DATA_FILE': BTC_DATA_FILE,
                 'usecols': ['Date', 'Closing Price (USD)', '24h Open (USD)', '24h High (USD)', '24h Low (USD)'],
                 'axes_map': [2,3,1,0]},
                {'currency': 'ETH',
                 'DATA_FILE': ETH_DATA_FILE,
                 'usecols': ['Date', 'Closing Price (USD)', '24h Open (USD)', '24h High (USD)', '24h Low (USD)'],
                 'axes_map': [2,3,1,0]}, 
                {'currency': 'LTC',
                 'DATA_FILE': LTC_DATA_FILE,
                 'usecols': ['date', 'close', 'high', 'low', 'open'],
                 'axes_map': [1,2,3,0]},
                {'currency': 'MKR',
                 'DATA_FILE': MKR_DATA_FILE,
                 'usecols': ['date', 'close', 'high', 'low', 'open'],
                 'axes_map': [1,2,3,0]}
                ]

NUM_CURRENCIES = len(LOADING_DICT)

## creates a save path for a model and its internal states
def create_save_paths(currency, order, train_frac, batch_size, hist_size, targ_size, overlap, learning_rate):
    specs = ''
    
    if   (MC == 0):
        specs += currency + '__'
    
    elif (MC == 1):
        for ENTRY in LOADING_DICT:
            specs += ENTRY['currency'] + '_'
        specs += '_'

    specs += 'ORDER_'+str(order)
    specs += '__'
    specs += 'TRAIN_FRAC_'+str(train_frac)
    specs += '__'
    specs += 'BATCH_SIZE_'+str(batch_size)
    specs += '__'
    specs += 'HIST_SIZE_'+str(hist_size)
    specs += '__'
    specs += 'TARG_SIZE_'+str(targ_size)
    specs += '__'
    specs += 'OVERLAP_'+str(overlap)
    specs += '__'
    specs += 'LR_'+str(learning_rate)
    specs += '__'

    if (VER == 2):
        specs += 'VER_2'
        specs += '__'

    model_save_path = './build_tools/'+specs+'lstm_model_params.pth'
    cell_save_path = './build_tools/'+specs+'lstm_model_cell.pth'
    hidden_save_path = './build_tools/'+specs+'lstm_model_hidden.pth'
    return model_save_path, cell_save_path, hidden_save_path

MODEL_SAVE_PATH, CELL_SAVE_PATH, HIDDEN_SAVE_PATH = create_save_paths(CURRENCY, ORDER, TRAIN_FRAC, BATCH_SIZE, HIST_SIZE, TARG_SIZE, OVERLAP, LR)
