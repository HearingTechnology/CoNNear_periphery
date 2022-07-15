## CoNNear: A convolutional neural-network model of the human auditory periphery

**A faster implementation of the CoNNear periphery model can be found under [branched](https://github.com/HearingTechnology/CoNNear_HAPM/tree/branched), which uses a branched version of the ANF model (https://doi.org/10.48550/arxiv.2207.07091).**

This repository contains the CoNNear periphery model and an example Python script `connear_example.py` that can be used to simulate the outputs of each different stage of the auditory periphery to basic auditory stimuli. The *connear* folder contains the five CoNNear modules, which correspond to the trained CNN models of each separate stage of auditory processing. The folder also contains four hearing-impaired (HI) cochlear models that were obtained via [transfer learning](http://dx.doi.org/10.21437/Interspeech.2020-2818), corresponding to the Flat25, Flat35, Slope25, Slope35 HI profiles of the [Verhulstetal2018](https://github.com/HearingTechnology/Verhulstetal2018Model) model. The CoNNear periphery model was first presented in [ICASSP 2022](https://doi.org/10.1109/ICASSP43922.2022.9747683) and was used to design DNN-based hearing-aid strategies. 

By default, the example script simulates and plots the outputs of each stage for a pure-tone stimulus of three different levels. The stimulus parameters can be adjusted in the *Simulation parameter definition* section to provide different inputs to the model. The CoNNear model can be executed on a CPU, but the execution time significantly improves with the use of a GPU in Tensorflow.

To simulate population responses across CF after the auditory-nerve (AN) level, the CN-IC python module `ic_cn2018.py` by [Verhulstetal2018 (v1.2)](https://github.com/HearingTechnology/Verhulstetal2018Model) is included and is used as backend. This repository also contains an `extra_functions.py` Python file with all supplementary functions, this `README.md` document and a license file. 

## How to test the CoNNear model

To run the example script and the CoNNear models in Python, Numpy, Scipy, Keras and Tensorflow are necessary. We used a conda environment (Anaconda v2020.02) that included the following versions: 
+ Python 3.6.1
+ Numpy v1.19.2
+ Scipy v1.5.4
+ Keras v2.3.1
+ Tensorflow v1.15.0

## CoNNear model specifications

The CoNNear model is comprised by five distinct AECNN models: One for the cochlea stage, one for the inner-hair-cell (IHC) stage and three for the three different auditory-nerve-fiber (ANF) types. For an auditory stimulus given as input, it predicts the basilar-membrane (BM) vibration, IHC transduction and ANF firing rates along 201 cochlear channels.
Each distinct model is included in the *connear* folder, with the json files corresponding to the Tensorflow model descriptions and the h5 files corresponding to the trained weights for each model. 1-CF json files are also included in the *connear* folder for each of the CoNNear IHC-ANF models which can be used separately to simulate single-unit neuronal responses. 
All models were trained on training sets comprised by 2310 speech sentences of the TIMIT speech dataset. Using the speech dataset as input to the reference model, the outputs of each stage were simulated and were used to form each training dataset. More details about the cochlear model can be found at https://doi.org/10.1038/s42256-020-00286-8 and about the IHC-ANF models at https://doi.org/10.1038/s42003-021-02341-5.

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

The ANF stage was adopted from the [CoNNear_IHC-ANF](https://github.com/HearingTechnology/CoNNear_IHC-ANF) repository and consists of three models, CoNNear<sub>ANfH</sub>, CoNNear<sub>ANfM</sub> and CoNNear<sub>ANfL</sub>, corresponding to high-spontaneous-rate (HSR), medium-spontaneous-rate (MSR) and low-spontaneous-rate (LSR) fibers, respectively. Each model consists of 28 layers, with 64 filters of 8 filter length each. A PReLU non-linearity is used after each layer of the CoNNear<sub>ANfH</sub> and CoNNear<sub>ANfM</sub>, while a combination of tanh and sigmoid non-linearities is used for the CoNNear<sub>ANfL</sub> model. All three models predict the firing rate of the respective ANF along the frequency channels given as input (by default 201 channels). The ANF model parameters are CF-independent, thus the 1-CF or 201-CF model descriptions can evenly be used. To avoid high memory requirements and to be able to simulate different numbers of frequency channels, it is suggested to use the single-unit ANF models. 

7936 context samples are added on the left side and 256 context samples are added on the right side of the input to the models. The ANF models can still take inputs of variable length, however, due to the increased number of layers, this sample length has to be a multiple of 16384 (2<sup>N<sub>enc</sub></sup>, where N<sub>enc</sub> = 14 is the number of layers in the encoder).

----
## Citation
If you use this code, please cite the corresponding papers:

**CoNNear<sub>cochlea</sub>**: Baby, D., Van Den Broucke, A. & Verhulst, S. A convolutional neural-network model of human cochlear mechanics and filter tuning for real-time applications. Nat Mach Intell 3, 134–143 (2021). https://doi.org/10.1038/s42256-020-00286-8

**CoNNear<sub>IHC-ANF</sub>**: Drakopoulos, F., Baby, D. & Verhulst, S. A convolutional neural-network framework for modelling auditory sensory cells and synapses. Commun Biol 4, 827 (2021). https://doi.org/10.1038/s42003-021-02341-5

**CoNNear<sub>ANF</sub> - branched**: Drakopoulos, F. & Verhulst, S. A neural-network framework for the design of individualised hearing-loss compensation. arXiv preprint arXiv:2207.07091 (2022). https://doi.org/10.48550/arxiv.2207.07091

This repository can also be cited separately:

Drakopoulos, F., Van Den Broucke, A., Baby, D. & Verhulst, S. HearingTechnology/CoNNear_periphery: CoNNear: A CNN model of the human auditory periphery (v1.0). Zenodo. (2022). [https://github.com/HearingTechnology/CoNNear_periphery](https://github.com/HearingTechnology/CoNNear_periphery).

[![DOI](https://zenodo.org/badge/322307161.svg)](https://zenodo.org/badge/latestdoi/322307161)

##
For questions, please reach out to one of the corresponding authors

* Fotios Drakopoulos: fotios.drakopoulos@ugent.be
* Arthur Van Den Broucke: arthur.vandenbroucke@ugent.be
* Deepak Baby: deepakbabycet@gmail.com
* Sarah Verhulst: s.verhulst@ugent.be

> This work was funded with support from the EU Horizon 2020 programme under grant agreement No 678120 (RobSpear).

