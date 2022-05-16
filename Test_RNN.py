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


HIST_SIZE           =            MODEL_CONSTANTS.HIST_SIZE
TARG_SIZE           =            MODEL_CONSTANTS.TARG_SIZE
OVERLAP             =              MODEL_CONSTANTS.OVERLAP
NUM_INPUT_CHANNELS  =   MODEL_CONSTANTS.NUM_INPUT_CHANNELS
NUM_OUTPUT_CHANNELS =  MODEL_CONSTANTS.NUM_OUTPUT_CHANNELS

BATCH_SIZE  =        MODEL_CONSTANTS.BATCH_SIZE

test_ds, _ = Aux.create_ds_and_dl()

parser = OptionParser()

parser.add_option("-t", "--testing", action="store_true", dest="testing", default=False,
                  help="restrict output times to existing dataset")


parser.add_option("-i", "--index", action="store", dest="idx",
                  help="specify a dataset index to propagate through the RNN")

parser.add_option("-o", "--surface_offset", action="store", dest="T",
                  help="generate a prediction surface with offset T")

parser.add_option("-s", "--save_states", action="store_true", dest="save", default=False,
                  help="save hidden and cell states at the end of data propagation")

(options, args) = parser.parse_args()

save = options.save

if (options.idx is not None):
    if (options.idx=='MAX'):
        if (options.testing):
            idx = len(test_ds)-1

        else:
            idx = len(test_ds)-1 + TARG_SIZE
        
    else:
        idx = int(options.idx)


"""

"""

lstm_models = []
lstm_model  = BPI_RNN.LSTM_Model(BATCH_SIZE, MODEL_CONSTANTS.NUM_INPUT_CHANNELS, HIST_SIZE, MODEL_CONSTANTS.NUM_OUTPUT_CHANNELS, TARG_SIZE)

for i in range(MODEL_CONSTANTS.ORDER+1):
    pth, c_pth, h_pth = MODEL_CONSTANTS.create_save_paths(MODEL_CONSTANTS.CURRENCY, MODEL_CONSTANTS.ORDER-i, MODEL_CONSTANTS.TRAIN_FRAC,
                                                          BATCH_SIZE, HIST_SIZE, TARG_SIZE, OVERLAP, MODEL_CONSTANTS.LR)
    
    lstm_model.load_state_dict(torch.load(pth))
    #lstm_model.lstm_op.load_states(torch.load(c_pth), torch.load(h_pth))
    lstm_models.append(lstm_model)
    
## reference prices for assessing network performance
ref_prices_pt = Aux.ref_prices_pt
final_ys, ref_idxs = None, None


if (options.T is not None):
    ref_idxs = None
    T = int(options.T)
    
    if (T > idx): print('Error: specified offset ({}) must be less than specified index ({})'.format(T, idx))
    
    idxs, predictions = [],[]
    
#########################################################################
##  propagate multiple test samples given by indices idx-T:idx
#########################################################################

    for t in range(T):
        with torch.no_grad():
            data = test_ds[idx-(T-t+1)]    
            x, y, y_idxs = data['x'], data['y'], data['y_idxs']
            x, y, y_idxs = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0), torch.unsqueeze(y_idxs, 0)
            
            y_pred = lstm_model(x)

            if (MODEL_CONSTANTS.ORDER == 1):
                y_cor = NLO_lstm_model(x)
                y_pred += y_cor

            # use y''[-1] because of extra batch dimension
            final_pred = y_pred[-1]
            ref_idxs = y_idxs[-1]
            final_y = Aux.integrate_output(ref_prices_pt, ref_idxs, final_pred)

            idxs.append(ref_idxs)
            predictions.append(final_y)

    ref_idxs = idxs[0][-T:].numpy()
    ref_prices = ref_prices_pt[:,ref_idxs]
    Aux.plot_prediction_surface(ref_prices_pt[:,ref_idxs], predictions, idxs, TARG_SIZE, 'Bitcoin Prediction Over Time', 'Network_PredictionSurface')
        
    
else:

#########################################################################
##  propagate a single test sample given by index idx
#########################################################################

    with torch.no_grad():
        #for i in range(len(test_ds), len(test_ds)+TARG_SIZE):
        for i in range(len(test_ds) + TARG_SIZE):
            data = test_ds.get_item_alt(i)
            
            x, _, y_idxs = data['x'], data['y'], data['y_idxs']
            x = torch.unsqueeze(x, 0)
            ref_idxs = np.arange(y_idxs[0]-TARG_SIZE, y_idxs[0])
                
            y_pred = torch.zeros_like(lstm_models[0](x))

            for model in lstm_models:
                y_pred += model(x)
                    
                final_pred = torch.squeeze(y_pred, 0)                    
                final_y = Aux.integrate_output(ref_prices_pt[:NUM_OUTPUT_CHANNELS], y_idxs, final_pred)
                    
                if save:
                    for i in range(MODEL_CONSTANTS.ORDER+1):
                        pth, c_pth, h_pth = MODEL_CONSTANTS.create_save_paths(MODEL_CONSTANTS.CURRENCY, MODEL_CONSTANTS.ORDER-i, MODEL_CONSTANTS.TRAIN_FRAC,
                                                                              BATCH_SIZE, HIST_SIZE, TARG_SIZE, OVERLAP, MODEL_CONSTANTS.LR)
                            
                        for model in lstm_models:
                            torch.save(model.state_dict(), pth)
                            torch.save(model.lstm_op.c, c_pth)            
                            torch.save(model.lstm_op.h, h_pth)
                            
        if (MODEL_CONSTANTS.MC == 0):
            # pass ref_prices_pt[:,1:] because training data is the relative change between timesteps
            LABELS = Aux.STANDARD_LABELS
                                    
            Aux.plot_data(ref_prices_pt[:,1:], final_y, ref_idxs, y_idxs, HIST_SIZE, TARG_SIZE, OVERLAP, Aux.dates,
                                  'Value Over Time', LABELS, 'Network_Test'+'_'+MODEL_CONSTANTS.CURRENCY)
                
        elif (MODEL_CONSTANTS.MC == 1):
            for i in range(MODEL_CONSTANTS.NUM_CURRENCIES):
                Aux.plot_data(ref_prices_pt[i,:,1:], final_y[i], ref_idxs[i], y_idxs[i], HIST_SIZE, TARG_SIZE, OVERLAP, Aux.dates,
                              'Value Over Time', MODEL_CONSTANTS.LOADING_DICT[i]['usecols'][1:], 'MC_Network_Test_'+MODEL_CONSTANTS.LOADING_DICT[i]['currency'])

            for i in range(len(test_ds) + TARG_SIZE):
                data = test_ds.get_item_alt(i)
                
                x, _, y_idxs = data['x'], data['y'], data['y_idxs']
                x = torch.unsqueeze(x, 0)
                ref_idxs = np.arange(y_idxs[0]-TARG_SIZE, y_idxs[0])
                
                y_pred = torch.zeros_like(lstm_models[0](x))
                
                for model in lstm_models:
                    y_pred += model(x)

                
                        
