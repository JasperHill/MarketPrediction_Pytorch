BPI_RNN
----
The efficacy of high-rank linear operators as described [here](https://github.com/JasperHill/Captcha_Tests_Pytorch/blob/master/Writeups/Writeup.pdf) is investigated within the context of\
a long-short-term memory(LSTM) layer trained to predict the bitcoin price index. The layout is similar to a standard LSTM layer. However, unlike traditional implementations, the configuration of this work lacks gate biases, and the weights, naturally, are high-rank linear operators.


## Model and hyperparameter initialization
The network is initially created via `BPI_RNN.py` under the `build_tools` subdirectory. This generates an untrained state dictionary. Also under `build_tools` are `Custom_RNNs.py` and `RNN_passes.cpp`, which define the classes and methods employed by the network. Lastly, the data file from which the training and testing sets are derived as well as the network configuration parameters are set in `MODEL_CONSTANTS.py`. These currently include:
* `TRAIN_FRAC`, which sets the fraction of the data to be used for training
* `OVERLAP`, which sets the amount of temporal overlap between the input data and the outputs
* `BATCH_SIZE`
* `HIST_SIZE`, which is the number of timesteps within each batch sample
* `TARG_SIZE`, which is the number of timesteps within each output sample
* `NUM_EPOCHS`
* `DATA_FILE`, which is the file under the `data` subdirectory on which the network is trained
* `MODEL_SAVE_PATH`, which is the save path for the model state dictionary

## Data preprocessing
The network takes inputs of shape `(BATCH_SIZE, 2, HIST_SIZE)` and yields outputs of shape `(BATCH_SIZE, 2, TARG_SIZE)` where the second tensor component corresponds to the high/low arrays. The program is written in such a way that it measures the size of each batch and resizes its hidden and cell state tensors accordingly. This is done so that datasets containing non-integer multiples of `BATCH_SIZE` elements can be handled properly. In lieu of absolute prices, which are intractably large, each element is converted to a relative price change. All preprocessing, postprocessing, and visualization machinery is defined in `Aux.py`, which, in turn, uses terms defined in `MODEL_CONSTANTS.py`. This sort of modularity simplifies high-level tweaking and enhances overal readability of this project.

## Training
The network is trained by calling `Train_RNN.py`. The flag `-c` will create a new, untrained model and save it to `MODEL_CONSTANTS.MODEL_SAVE_PATH` after training. Otherwise, the script will assume a model already exists at that path, load it, and train it. After `NUM_EPOCHS` training epochs have been completed, a sample from the testing dataset is propagated through the network, and the output is integrated and plotted alongside true market prices, saved as `Network_Prediction.pdf`. Also saved are graphs of the mean squared error (MSE) and the percent error of each prediction element as `RNN_MSE.pdf` and `RNN_PE.pdf` respectively. Finally, the trained network state dictionary is saved. 

## Testing
An arbitrary sequence of testing data may be propagated through a pretrained RNN by running `Test_RNN.py --index=idx` where `idx` is the dataset index to be propagated. The result will be plotted alongside real data and saved to `Network_Test.pdf`. Note also that the beginning timestep of the output will be offset by `MODEL_CONSTANTS.HIST_SIZE - MODEL_CONSTANTS.OVERLAP`.