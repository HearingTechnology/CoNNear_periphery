## CoNNear: A convolutional neural-network model of the human auditory periphery

This repository contains the branched version of the CoNNear periphery model and an example Python script `connear_example.py` that can be used to simulate the outputs of each different stage of the auditory periphery to basic auditory stimuli. The *connear* folder contains the three CoNNear modules, which correspond to the trained CNN models of each separate stage of auditory processing. The folder also contains four hearing-impaired (HI) cochlear models obtained via [transfer learning](http://dx.doi.org/10.21437/Interspeech.2020-2818), corresponding to the Flat25, Flat35, Slope25, Slope35 HI profiles of the [Verhulstetal2018](https://github.com/HearingTechnology/Verhulstetal2018Model) model. The CoNNear periphery model was first presented in [ICASSP 2022](https://doi.org/10.1109/ICASSP43922.2022.9747683) and was used to design DNN-based hearing-aid strategies.

By default, the example script simulates and plots the outputs of each stage for a pure-tone stimulus of three different levels. The stimulus parameters can be adjusted in the *Simulation parameter definition* section to provide different inputs to the model. The CoNNear model can be executed on a CPU, but the execution time significantly improves with the use of a GPU in Tensorflow.

To simulate population responses across CF after the auditory-nerve (AN) level, the CN-IC python module `ic_cn2018.py` by [Verhulstetal2018 (v1.2)](https://github.com/HearingTechnology/Verhulstetal2018Model) is included and is used as backend. This repository also contains an `extra_functions.py` Python file with all supplementary functions, this `README.md` document and a license file. 

## How to test the CoNNear model

To run the example script and the CoNNear models in Python, Numpy, Scipy, Keras and Tensorflow are necessary. We used a conda environment (v4.10.3) that included the following versions: 
+ Python 3.8.5
+ Numpy v1.20.3
+ Scipy v1.4.1
+ Keras v2.8.0
+ Tensorflow v2.8.0

## CoNNear model specifications

The CoNNear model is comprised by three distinct encoder-decoder CNN models: One for the cochlea stage, one for the inner-hair-cell (IHC) stage and one for the three different auditory-nerve-fiber (ANF) types. For an auditory stimulus given as input, it predicts the basilar-membrane (BM) vibration, IHC transduction and ANF firing rates along 201 cochlear channels.
Each distinct model is included in the *connear* folder, with the json files corresponding to the Tensorflow model descriptions and the h5 files corresponding to the trained weights for each model. 1-CF json files are also included in the *connear* folder for each of the CoNNear IHC-ANF models which can be used separately to simulate single-unit neuronal responses. 
All models were trained on training sets comprised by 2310 speech sentences of the TIMIT speech dataset. Using the speech dataset as input to the reference model, the outputs of each stage were simulated and were used to form each training dataset. More details about the cochlear model can be found at https://doi.org/10.1038/s42256-020-00286-8 and about the IHC-ANF models at https://doi.org/10.1038/s42003-021-02341-5. A faster implementation of the ANF model is used here, which was presented in our recent [paper]() and was used for the optimization of DNN-based hearing-aid strategies.

#### Cochlea stage

The CoNNear<sub>cochlea</sub> model was adopted from the [CoNNear_cochlea](https://github.com/HearingTechnology/CoNNear_cochlea) repository and consists of 8 layers, each with 128 filters of 64 filter length and a tanh non-linearity.
For a given auditory stimulus, CoNNear<sub>cochlea</sub> predicts the BM vibration waveforms along 201 cochlear channels, resembling a frequency range from 122Hz to 12kHz based on the Greenwood map. 
256 context samples are added on both sides to account for possible loss of information at the boundaries. 

The CoNNear cochlea model can take a stimulus with a variable time length as an input, however, due to the convolutional character of CoNNear, the length has to be a multiple of 16 (2<sup>N<sub>enc</sub></sup>, where N<sub>enc</sub> = 4 is the number of layers in the encoder).

#### IHC stage

The CoNNear<sub>IHC</sub> model was adopted from the [CoNNear_IHC-ANF](https://github.com/HearingTechnology/CoNNear_IHC-ANF) repository and consists of 6 layers, each with 128 filters of 16 filter length. A tanh non-linearity is used after each layer in the encoder and a sigmoid non-linearity after each layer of the decoder.
For BM vibration inputs, CoNNear<sub>IHC</sub> predicts the IHC receptor potential along the same frequency channels (by default 201 cochlear channels). 
256 samples of context are added at both sides. 

The CoNNear IHC model can take a stimulus with a variable time length as an input, however, the length has to be a multiple of 8 (2<sup>N<sub>enc</sub></sup>, where N<sub>enc</sub> = 3 is the number of layers in the encoder). The IHC model parameters are CF-independent, thus the 1-CF or 201-CF model descriptions can be evenly used.

#### ANF stage

The CoNNear<sub>ANF</sub> model is a branched encoder-decoder model, comprised by one shared encoder and three decoders that simulate the outputs of high-spontaneous-rate (HSR), medium-spontaneous-rate (MSR) and low-spontaneous-rate (LSR) fibers. The model uses 14 layers in the encoder and in each decoder, with 64 filters of 8 filter length and a PReLU non-linearity for each layer. More information about the architecture can be found [here]().
For IHC receptor potential inputs, CoNNear<sub>ANF</sub> predicts the firing rate of the respective ANFs along the frequency channels (by default 201). The ANF model parameters are CF-independent, thus the 1-CF or 201-CF model descriptions can be evenly used. Although the 201-CF models can simulate ANF responses for the 201 channels in parallel, the single-unit ANF models can be used for the simulation of less CFs or if high memory allocation is encountered with the 201-CF models.

7936 context samples are added on the left side and 256 context samples are added on the right side of the input to the models. The ANF models can still take inputs of variable length, however, due to the increased number of layers, this sample length has to be a multiple of 16384 (2<sup>N<sub>enc</sub></sup>, where N<sub>enc</sub> = 14 is the number of layers in the encoder).

----
For questions, please reach out to one of the corresponding authors

* Fotios Drakopoulos: fotios.drakopoulos@ugent.be
* Arthur Van Den Broucke: arthur.vandenbroucke@ugent.be
* Deepak Baby: deepakbabycet@gmail.com
* Sarah Verhulst: s.verhulst@ugent.be

> This work was funded with support from the EU Horizon 2020 programme under grant agreement No 678120 (RobSpear).

