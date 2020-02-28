BPI_RNN
----
The efficacy of high-rank linear operators as described [here](https://github.com/JasperHill/Captcha_Tests_Pytorch/blob/master/Writeups/Writeup.pdf) is investigated within the context of\
a long-short-term memory(LSTM) layer trained to predict the bitcoin price index. The layout is similar to a standard LSTM layer. However, unlike traditional implementations, the configuration of this work lacks gate biases, and the weights, naturally, are high-rank linear operators.

The network is created via `BPI_RNN.py` under the `build_tools` subdirectory. This generates an untrained state dictionary, which is accessed and overwritten by executing `Train_RNN.py`. Also under `build_tools` are `Custom_RNNs.py` and `RNN_passes.cpp`, which define the classes and methods employed by the network. Lastly, the data file from which the training and testing sets are derived as well as the network configuration parameters are set in `MODEL_CONSTANTS.py`. These currently include:
* `BATCH_SIZE`
* `HIST_SIZE`, which is the number of timesteps within each batch sample
* `TARG_SIZE`, which is the number of timesteps within each output sample
* `DATA_FILE`, which is the file under the `data` subdirectory on which the network is trained

The network, then, takes inputs of shape `(BATCH_SIZE, 2, HIST_SIZE)` and yields outputs of shape `(BATCH_SIZE, 2, TARG_SIZE)`. However, the program is written in such a way that it measures the size of each batch and resizes its hidden and cell state tensors accordingly. This is done so that datasets containing non-integer multiples of `BATCH_SIZE` elements can be handled properly. Also, note that the second shape index corresponds to the high and low prices.

By default, the datasets are constructed so that there is zero temporal overlap between the input and output tensors, but this can be modified by setting `OVERLAP` in `Train_RNN.py` to a nonzero integer. One could also specify a negative integer to produce a time delay between the last input element and the first output element, though this seems illogical and quixotic.

The final important variables are `train_frac` and `EPOCHS`, which specify the fraction of data to be used for training and the number of training epochs for the program to execute respectively.
