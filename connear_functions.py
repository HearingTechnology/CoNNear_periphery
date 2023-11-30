# -*- coding: utf-8 -*-
"""
This script contains all the supplementary functions needed to execute the
CoNNear example script in python.

@author: Fotios Drakopoulos, UGent, Feb 2022
"""

import numpy as np
import tensorflow as tf
import scipy.signal as sp_sig
import scipy.io.wavfile
from typing import List, Optional, Tuple, Union
from os import path

from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

def rms (x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute RMS energy of a matrix over the specified axis."""
    sq = np.mean(np.square(x), axis = axis)
    return np.sqrt(sq)

def next_power_of_2(x: int) -> int:
    """Compute the next power of 2 bigger than the input x."""
    return 1 if x == 0 else int(2**np.ceil(np.log2(x)))

def wavfile_read(wavfile: str,fs: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Read a wavfile and normalize it to +/-1.
    If fs is given the signal is resampled to the given sampling frequency.
    """
    fs_signal, speech = scipy.io.wavfile.read(wavfile)
    if not fs:
        fs=fs_signal

    if speech.dtype != 'float32' and speech.dtype != 'float64':
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

def load_connear_model(modeldir: str, json_name: str = "/Gmodel.json",
                       weights_name: str =  "/Gmodel.h5", 
                       crop: bool = True, name: Optional[str] = None) -> tf.keras.Model:
    """Function to load one portion of a CoNNear model.
    
    Args:
      modeldir: Where to find the model and weight files
      json_name: The file within modeldir to find the TF model description
      weights_name: The file within modeldir to find the weights data
      crop: Drop the last layer of the model, which crops out the extra context 
        provided to the model.  Set to False to remove this last layer and provide 
        the context to the next part of the model.
      name: The name for this new model.

    Returns:
      A Tensorflow Model
    """
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

    return model
    
def build_connear(modeldir: str, poles: str = '', Ncf: int = 201, full_model: bool = False,
                  name: str = 'periphery_model', print_summary: bool = False) -> Union[
                      Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model],
                      Tuple[tf.keras.Model, tf.Tensor]]:
    """Function to load the separate pretrained CoNNear modules.

    Args:
      modeldir: From which directory to load the model and weights
      poles: The HL curve to model. If a string is given then the corresponding HI weights are 
        loaded for the cochlear model. The default is to model a normal auditory system. 
        (The poles name was adopted from the Verhulst et al model as it was used to define the 
        different cochlear filter profiles/tuning.)
      Ncf: Number of channels (center frequencies). If no argument is given for the Ncf 
        variable then the 201-CF CoNNear models are loaded, otherwise the function tries to 
        load the CoNNear model definitions with the number of channels given in Ncf
      full_model: Whether to return three separate periphery models (cochlea, IHC, ANFs) or 
        a (combined) full periphery model. By default the function loads and returns each
        module separately.
      name: The name assigned to this Model.
      print_summary: Print the Keras summary of the model
    """
    assert(Ncf == 201 or Ncf == 21 or Ncf == 1), "Only 201-, 21- and 1-CF models are provided by the current framework for the moment."
    if Ncf != 201:
        cf_flag = '_' + str(Ncf) + 'cf'
    else:
        cf_flag = ''
    # Cochlea
    poles = poles.lower() # make lowercase
    if (not poles) or poles == 'nh':
        cochlea = load_connear_model(modeldir,json_name="/cochlea.json",weights_name="/cochlea.h5",name="cochlea_model",crop=0)
    else:
        assert(path.exists(modeldir + "/cochlea_" + poles + ".h5")), "The poles for the selected HI profile do not exist. HI cochlear models are available for the following hearing-loss profiles: Flat25, Flat35, Slope25, Slope35"
        cochlea = load_connear_model(modeldir,json_name="/cochlea.json",weights_name="/cochlea_" + poles + ".h5",name="cochlea_model",crop=0)
    cochlea.trainable=False
    for l in cochlea.layers:
        l.trainable=False
    # IHC
    ihc = load_connear_model(modeldir,json_name="/ihc" + cf_flag + ".json",weights_name="/ihc.h5",name="ihc_model",crop=0)
    ihc.trainable=False
    for l in ihc.layers:
        l.trainable=False
    # ANF
    anf = load_connear_model(modeldir,json_name="/anf" + cf_flag + ".json",weights_name="/anf.h5",name="anf_model")
    anf.trainable=False
    for l in anf.layers:
        l.trainable=False

    if full_model:
        audio_in = Input(shape=(None,1), name="audio_input", dtype='float32')
        cochlea = Model(cochlea.layers[0].input,cochlea.layers[-1].output)
        cochlea_out = cochlea(audio_in)
        if Ncf < 201 and Ncf > 1: # downsample the frequency channels of CoNNear cochlea
            CFs = K.arange(0,201,int((201-1)/(Ncf-1)))
            cochlea_out=Lambda(lambda x: tf.gather(x,CFs,axis=2), name='freq_downsampling')(cochlea_out)
        cochlea = Model(audio_in, cochlea_out)
        # IHC
        ihc = ihc(cochlea.layers[-1].get_output_at(-1))
        # ANF
        anf = anf(ihc)
        # periphery model
        periphery = Model(inputs=cochlea.layers[0].input,outputs=anf,name=name)
        periphery.layers[-3].name = 'cochlea_model'
        periphery.layers[-2].name = 'ihc_model'
        periphery.layers[-1].name = 'anf_model'
        if print_summary:
            periphery.summary()

        return periphery, CFs
    else:
        return cochlea, ihc, anf

def slice_1dsignal(signal: np.ndarray, window_size: int, winshift: int, minlength: int, 
                   left_context:int = 256, right_context:int = 256) -> np.ndarray:
    """Return windows of the given signal by sweeping in stride fractions of window. Slices that are
    less than minlength are omitted. Input signal must be a 1D-shaped array.

    Args:
      signal: A one-dimensiona input waveform
      window_size: The size of each window of data from the signal.
      winshift: How much to shift the window as we progress down the signal. 
        If winshift = window_size then the overlap between windows is 0.
      minlength: Drop (final) windows that have less than this number of samples.
      left_context: How much context to add (from earlier parts of the signal) before
        the current window. (Or add zeros if not enough signal)
      right_context: Like left, but to the right of the current window.
    
    Returns:
      A 3D tensor of size num_frames x window_size x 1
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

        slices.append(slice_)
    slices = np.vstack(slices)
    slices = np.expand_dims(slices, axis=2) # the CNN will need 3D data
    return slices
    
def unslice_3dsignal(signal: np.ndarray, winlength: int, winshift: int, 
                     ignore_first_set: int = 0, fs: float = 20e3, 
                     trailing_silence: float = 0.) -> np.ndarray:
    """Reconstructs an 1-D signal sliced in windows. Merge the different windows of 
    the signal, undoing the slice format above, usually after processing. Overlap 
    and add the different frames.

    Args:
      signal: A 3d tensor of shape num_frames x winlength x num_channels
      winlength: The size of each window of data from the signal with context on
      both sides (window_size + left_context + right_context)
      winshift: How much each window of data is shifted in samples along the signal,
      e.g. if winshift = winlength then the overlap between windows is 0.
      ignore_first_set: Set of samples that will be omitted from the beginning of the
      provided signal, i.e. context in the sliced signal. This parameter can be used 
      to skip the left context from the sliced signal when merging consecutive windows.
      This argument should be used in combination with the trailing_silence parameter
      to remove left context present in all windows of the sliced signal. However, 
      it is suggested to manually remove context from the sliced array before providing
      it as an input to this function and setting these parameters to 0 instead, as 
      proper functionality is not guaranteed.
      fs: Sample rate of the data, needed for scaling trailing silence (if provided).
      trailing_silence: Length of zeros (silence) to be removed from the beginning of
      the reconstructed signal, if context was present originally. Defined in s and 
      scaled based on the fs.
    
    Returns:
      A numpy array of size 1 x number of samples x number of channels.
    """
    assert len(signal.shape) == 3, "signal must be a 3D-shaped array"

    nframes = signal.shape[0]
    slength = ((nframes - 1)) * winshift + winlength # total length of the reconstructed signal
    tl_2d = np.zeros((slength, signal.shape[2]))
    scale_ = np.zeros((slength,1))  # Keep track of number of windows added here.
    dummyones = np.ones((signal.shape[0], signal.shape[1]))
    trailing_zeros = int(trailing_silence * fs) # zeros to add at the beginning of the reconstructed signal
    # Add in the first frame.
    sigrange = range (winlength)
    tl_2d [sigrange, :] = tl_2d [sigrange, :] + signal[0]
    scale_[sigrange,0] = scale_[sigrange,0] + dummyones[0]
    # Overlap-add
    for i in range(1,nframes):
        sigrange = range (i * winshift + ignore_first_set, (i*winshift) + winlength)
        tl_2d [sigrange, :] = tl_2d [sigrange, :] + signal[i,ignore_first_set:,:]
        scale_[sigrange,0] = scale_[sigrange,0] + dummyones[i,ignore_first_set:]
    # Scale the overlapping parts accordingly
    tl_2d /= scale_
    # Return the reconstructed signal as a 3-D array
    tl_2d = np.expand_dims(tl_2d[trailing_zeros:,:], axis=0)
    return tl_2d
 
def compute_oae(vbm_out: np.ndarray, cf_no: int = 0,sig_start: int = 0) -> Tuple[np.array, int]:
    """Compute the fft of the vbm output over the cf_no channel to predict the oae
    The fft is applied on the second dimension (axis=1).

    Args:
      vbm_out: The velocity of the basilar membrane motion.
        An array of size num_frames x frame_size x num_channels
      cf_no: Which channel number to pick out
      sig_start: How many samples to skip from the beginning of the frame, to avoid 
        including any context or silence in the FFT estimation.
    
    Returns:
      A 2-ple containing: 
        oae_fft_mag: the OAE magnitude of size num_frames x fftSize/2+1
        nfft: the size of the real part of the fft.
    """
    oae_sig = vbm_out[:, sig_start:, cf_no] # pick a CF
    oae_fft = np.fft.fft(oae_sig)
    nfft = int(oae_fft.shape[1]/2+1)
    oae_fft_mag = np.absolute(oae_fft[:,:nfft])
    return oae_fft_mag, nfft
