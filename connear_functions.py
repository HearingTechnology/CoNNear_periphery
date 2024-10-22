# -*- coding: utf-8 -*-
"""
This script contains all the supplementary functions needed to execute the
CoNNear example script in python.

@author: Fotios Drakopoulos, UGent, 12/2020
"""

import numpy as np
import tensorflow as tf
import scipy.signal as sp_sig
import scipy.io.wavfile
from os import path

from keras.models import model_from_json, Model, Input
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

def rms (x, axis=0):
    # compute rms of a matrix
    sq = np.mean(np.square(x), axis = axis)
    return np.sqrt(sq)
    
def next_power_of_2(x):
    return 1 if x == 0 else int(2**np.ceil(np.log2(x)))

def wavfile_read(wavfile,fs=[]):
    # read a wavfile and normalize it
    # if fs is given the signal is resampled to the given sampling frequency
    fs_signal, speech = scipy.io.wavfile.read(wavfile)
    if not fs:
        fs=fs_signal

    if speech.dtype != 'float32':
        if speech.dtype == 'int16':
            nb_bits = 16 # -> 16-bit wav files
        elif speech.dtype == 'int32':
            nb_bits = 32 # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        speech = speech / (max_nb_bit + 1.0) # scale the signal to [-1.0,1.0]

    if fs_signal != fs :
        signalr = sp_sig.resample_poly(speech, fs, fs_signal)
    else:
        signalr = speech

    return signalr, fs

def load_connear_model(modeldir,json_name="/Gmodel.json",weights_name="/Gmodel.h5",crop=1,name=[]):
    # Function to load each CoNNear model using tensorflow and keras
    #print ("loading model from " + modeldir )
    json_file = open (modeldir + json_name, "r")
    loaded_model_json = json_file.read()
    json_file.close()

    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = model_from_json(loaded_model_json, custom_objects={'tf': tf})
    if name:
        try:
            model.name = name
        except: # fix tensorflow 2 compatibility
            model._name = name
    model.load_weights(modeldir + weights_name)
    
    if not crop: # for connecting the different modules
        model=model.layers[1]
        if name:
            model=Model(model.layers[0].input, model.layers[-2].output,name=name)
        else:
            model=Model(model.layers[0].input, model.layers[-2].output) # get uncropped output
    #else:
    #    model=Model(model.layers[0].input, model.layers[-1].output) # get cropped output

    return model
    
def build_connear(modeldir,poles='',cf_flag='',full_model=0):
    ## Function to load the separate CoNNear modules
    ## If a string is given for poles then the corresponding HI weights are
    ## loaded for the cochlear model
    ## If 1 is given for the cf_flag then the 1-CF CoNNear models are loaded
    ## If 1 is given for the full_model argument then a concatenated version 
    ## of the full periphery model (cochlea, IHC, ANFs) is returned 
    ## By default the function loads and returns each module separately
    if cf_flag:
        cf_flag = '_1cf'
    # Cochlea
    poles = poles.lower() # make lowercase
    if (not poles) or poles == 'nh':
        cochlea = load_connear_model(modeldir,json_name="/cochlea.json",weights_name="/cochlea.h5",name="cochlea_model",crop=0)
    else:
        assert(path.exists(modeldir + "cochlea_" + poles + ".h5")), "The poles for the selected HI profile do not exist. HI cochlear models are available for the following hearing-loss profiles: Flat25, Flat35, Slope25, Slope35"
        cochlea = load_connear_model(modeldir,json_name="/cochlea.json",weights_name="/cochlea_" + poles + ".h5",name="cochlea_model",crop=0)
    # IHC
    ihc = load_connear_model(modeldir,json_name="/ihc" + cf_flag + ".json",weights_name="/ihc.h5",name="ihc_model",crop=0)
    # ANFs
    anfh = load_connear_model(modeldir,json_name="/anfh" + cf_flag + ".json",weights_name="/anfh.h5",name="anfh_model")
    anfm = load_connear_model(modeldir,json_name="/anfm" + cf_flag + ".json",weights_name="/anfm.h5",name="anfm_model")
    anfl = load_connear_model(modeldir,json_name="/anfl" + cf_flag + ".json",weights_name="/anfl.h5",name="anfl_model")
    
    if full_model:
        audio_in =  Input(shape=(None,1), name="audio_input", dtype='float32')
        cochlea = Model(cochlea.layers[0].input,cochlea.layers[-1].output)
        cochlea_out = cochlea(audio_in)
        cochlea = Model(audio_in, cochlea_out)
        # IHC
        ihc = ihc(cochlea.layers[-1].get_output_at(-1))
        # ANFs
        anfh = anfh(ihc)
        anfm = anfm(ihc)
        anfl = anfl(ihc)
        # IHC-AN complex model
        periphery = Model(input=cochlea.layers[0].input,output=[anfh,anfm,anfl])
        periphery.layers[-5].name = 'cochlea_model'
        periphery.layers[-4].name = 'ihc_model'
        periphery.layers[-3].name = 'anfh_model'
        periphery.layers[-2].name = 'anfm_model'
        periphery.layers[-1].name = 'anfl_model'
        #periphery.summary()
        
        return periphery
        
    else:
        return cochlea, ihc, anfh, anfm, anfl
    
def slice_1dsignal(signal, window_size, winshift, minlength, left_context=256, right_context=256):
    """ 
    Return windows of the given signal by sweeping in stride fractions of window.
    Slices that are less than minlength are omitted.
    Signal must be a 1D-shaped array.
    """
    assert len(signal.shape) == 1, "signal must be a 1D-shaped array"
    
    # concatenate zeros to beginning for adding context
    n_samples = signal.shape[0]
    num_slices = (n_samples)
    slices = [] # initialize empty array 

    for beg_i in range(0, n_samples, winshift):
        beg_i_context = beg_i - left_context
        end_i = beg_i + window_size + right_context
        if n_samples - beg_i < minlength :
            break
        if beg_i_context < 0 and end_i <= n_samples:
            slice_ = np.concatenate((np.zeros((1, left_context - beg_i)),np.array([signal[:end_i]])), axis=1)
        elif end_i <= n_samples: # beg_i_context >= 0
            slice_ = np.array([signal[beg_i_context:end_i]])
        elif beg_i_context < 0: # end_i > n_samples
            slice_ = np.concatenate((np.zeros((1, left_context - beg_i)),np.array([signal]), np.zeros((1, end_i - n_samples))), axis=1)
        else :
            slice_ = np.concatenate((np.array([signal[beg_i_context:]]), np.zeros((1, end_i - n_samples))), axis=1)
        #print(slice_.shape)
        slices.append(slice_)
    slices = np.vstack(slices)
    slices = np.expand_dims(slices, axis=2) # the CNN will need 3D data
    return slices
    
def unslice_3dsignal(signal, winlength, winshift, ignore_first_set=0, fs = 20e3, trailing_silence = 0.):
    """ 
    Merge the different windows of the signal.
    The first dimension corresponds to the windows and the second to the samples of each window.
    """
    assert len(signal.shape) == 3, "signal must be a 3D-shaped array"
    
    nframes = signal.shape[0]
    slength = ((nframes - 1)) * winshift + winlength
    tl_2d = np.zeros((slength, signal.shape[2]))
    scale_ = np.zeros((slength,1))
    dummyones = np.ones((signal.shape[0], signal.shape[1]))
    trailing_zeros = int(trailing_silence * fs)
    sigrange = range (winlength)
    tl_2d [sigrange, :] = tl_2d [sigrange, :] + signal[0]
    scale_[sigrange,0] = scale_[sigrange,0] + dummyones[0]
    for i in range(1,nframes):
        sigrange = range (i * winshift + ignore_first_set, (i*winshift) + winlength)
        tl_2d [sigrange, :] = tl_2d [sigrange, :] + signal[i,ignore_first_set:,:]
        scale_[sigrange,0] = scale_[sigrange,0] + dummyones[i,ignore_first_set:]
    
    tl_2d /= scale_
    tl_2d = np.expand_dims(tl_2d[trailing_zeros:,:], axis=0)
    return tl_2d
    
def compute_oae(vbm_out, cf_no=0,sig_start=0):
    # compute the fft of the vbm output over the cf_no channel to predict the oae
    # the fft is applied on the second dimension (axis=1)
    oae_sig = vbm_out[:, sig_start:, cf_no] # pick a CF
    oae_fft = np.fft.fft(oae_sig)
    nfft = int(oae_fft.shape[1]/2+1)
    oae_fft_mag = np.absolute(oae_fft[:,:nfft])
    return oae_fft_mag, nfft
