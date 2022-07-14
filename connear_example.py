# -*- coding: utf-8 -*-
"""
This example script simulates the outputs of CoNNear, a CNN-based model of the 
auditory periphery, in response to a pure-tone stimulus for 3 different levels.
The stimulus can be adjusted to simulate different frequencies, levels or 
different types of stimulation (clicks, SAM tones).

CoNNear is consisted of three distinct modules, corresponding to the cochlear
stage, the IHC stage and the AN stage of the auditory periphery. The sampling
frequency of all CoNNear modules is 20 kHz. The CoNNear cochlea model can be 
substituted by one of the pre-trained hearing-impaired models to simulate the 
responses of a periphery with cochlear gain loss. Otoacoustic emissions (OAEs) 
can also be simulated from the cochlear outputs. Cochlear synaptopathy can also 
be simulated in the AN stage by adapting the number of AN fibers. 

To simulate population responses across CF after the AN level, the CN-IC python
module by Verhulstetal2018 (v1.2) is used as backend. The response waves I, 
III and V can be extracted to simulate auditory brainstem responses (ABRs) or 
envelope following responses (EFRs). 

The execution of CoNNear in Tensorflow can be significantly sped up with the 
use of a GPU.

@author: Fotios Drakopoulos, UGent, Feb 2022
"""

from connear_functions import *
import matplotlib.pyplot as plt
import ic_cn2018 as nuclei
from time import time

tf.compat.v1.disable_eager_execution() # speeds up execution in Tensorflow v2

#################### Simulation parameter definition #########################
# Define the pure tone stimulus
f_tone = 1e3 # frequency of the pure tone
L = np.array([30.,70.,90.]) # levels in dB SPL to simulate
stim_dur = 400e-3 # duration of the stimulus
initial_silence = 5e-3 # silence before the onset of the stimulus
win_dur = 5.0e-3 # 5ms long hanning window for gradual onset
p0 = 2e-5 # calibrate to 2e-5 Pascal
# fmod = 98 # modulation frequency - uncomment for SAM tone stimuli
# m = 0.85  # modulation depth - uncomment for SAM tone stimuli

# Change the poles variable to include HI in the cochlear stage
poles = '' # choose between NH, Flat25, Flat35, Slope25, Slope35
# Simulate population response waves I, III, V & EFR
population_sim = 1 # set to 0 to omit the CN-IC stage
# Change the number of [HSR, MSR, LSR] ANFs to include cochlear synaptopathy for the population responses
num_anfs = [13,3,3] # 13,3,3 is the NH innervation, change to lower values (e.g. 13,0,0 or 10,0,0) to simulate cochlear synaptopathy
# Pick a channel number for which to plot the single-unit responses
No = 122 # between 0 and 200 (201 CFs are used for CoNNear) - 122 corresponds to ~1 kHz
# Simulate otoacoustic emissions (increases computation time)
oae_sim = 0 # set to 1 to generate and plot OAEs from the cochlear output

# number of parallely executed CFs for the IHC and ANF models
Ncf = 201 # set to 1 if the execution is too slow

#################### Main script #############################################
# Model-specific variables
fs = 20e3
context_left = 7936 # samples (396.8 ms) - left context of the ANF model
context_right = 256 # samples (12.8 ms) - right context 
# load model CFs 
CF = np.loadtxt('connear/cf.txt')*1e3
# scaling applied to the CoNNear outputs to bring them back to the original representations
cochlea_scaling = 1e6
ihc_scaling = 1e1
an_scaling = 1e-2
# CoNNear model directory
modeldir = 'connear/'
# number of layers in the deepest architecture used (ANF model) - for padding the input accordingly
Nenc = 14
# OAE parameters - shifted windows of the stimulus are used to get a smooth frequency representation
oae_cf_no = 0 # use the highest frequency channel (~12 kHz)
oae_window = 4096 # use a smaller window to generate shifted slices of the full stimulus
oae_step = 50 # the step with which the window is shifted - decrease to get smoother responses (longer simulation times)
cochlea_context = 256 # the cochlear model requires 256 samples of context on each side of the input

# Make stimulus
t = np.arange(0., stim_dur, 1./fs)
#stim_sin = np.ones(t.shape) # uncomment for click stimuli
stim_sin = np.sin(2 * np.pi * f_tone * t) # uncomment for pure-tone stimuli
#stim_sin = (1 + m * np.cos(2 * np.pi * fmod * t)) * np.sin(2 * np.pi * f_tone * t) # uncomment for SAM tone stimuli
# apply hanning window
if win_dur:
    winlength = int(2*win_dur * fs)
    win = sp_sig.windows.hann(winlength) # double-sided hanning window
    stim_sin[:int(winlength/2)] = stim_sin[:int(winlength/2)] * win[:int(winlength/2)]
    stim_sin[-int(winlength/2):] = stim_sin[-int(winlength/2):] * win[int(winlength/2):]
total_length = context_left + int(initial_silence * fs) + len(stim_sin) + context_right
stim = np.zeros((len(L), total_length))
stimrange = range(context_left + int(initial_silence * fs), context_left + int(initial_silence * fs) + len(stim_sin))
for i in range(len(L)):
    stim[i, stimrange] = p0 * 10**(L[i]/20) * stim_sin / rms(stim_sin) # calibrate
stim = np.expand_dims(stim, axis=2) # make the stimulus 3D

# Check the stimulus time-dimension size
if stim.shape[1] % 2**Nenc: # input size needs to be a multiple of 16384 for the ANF model
    Npad = int(np.ceil(stim.shape[1]/(2**Nenc)))*(2**Nenc)-stim.shape[1]
    stim = np.pad(stim,((0,0),(0,Npad),(0,0))) # zero-pad the 2nd dimension (time)

# Load each CoNNear module separately
cochlea, ihc, anf = build_connear(modeldir,poles=poles,Ncf=Ncf) # load the CoNNear models

print('CoNNear: Simulating auditory periphery stages')
time_elapsed=time()

# Cochlea stage
vbm = cochlea.predict(stim) # BM vibration
# IHC-ANF stage
if Ncf == 201: # use the 201-CF models
    vihc = ihc.predict(vbm) # IHC receptor potential
    ranf = anf.predict(vihc) # ANF firing rate
else: # use the 1-CF IHC-ANF models
    vihc = np.zeros(vbm.shape)
    ranf = np.zeros((3,vihc.shape[0],vihc.shape[1]-context_left-context_right,vihc.shape[2]))
    for cfi in range (0,CF.size): # simulate one CF at a time to avoid memory issues
        vihc[:,:,[cfi]] = ihc.predict(vbm[:,:,[cfi]])
        ranf[:,:,:,[cfi]] = anf.predict(vihc[:,:,[cfi]])

time_elapsed = time() - time_elapsed
print('Simulation finished in ' + '%.2f' % time_elapsed + ' seconds')

# Simulate otoacoustic emissions
if oae_sim:
    print('CoNNear: Simulating otoacoustic emissions')
    oae_size = oae_window # size of the oae response to keep 
    oae_min_window = oae_window-oae_step-int((initial_silence+win_dur)*fs) # minimum length of slice to include 
    oae = np.zeros((vbm.shape[0],oae_size,vbm.shape[2])) # pre-allocate array
    for li in range(0,L.size):
        # produce shifted versions of the input signal to get smoother OAEs
        stim_oae = stim[li,context_left:-context_right,0] # use the stimulus without context
        stim_oae_slices = slice_1dsignal(stim_oae, oae_window, oae_step, oae_min_window, left_context=cochlea_context, right_context=cochlea_context) # 256 samples of context are added on the sides
        vbm_oae_slices = cochlea.predict(stim_oae_slices) # simulate the outputs for the generated windows
        vbm_oae_slices = vbm_oae_slices[:,cochlea_context:-cochlea_context,:] # remove the context from the cochlear outputs
        # undo the windowing to get back the full response
        vbm_oae = unslice_3dsignal(vbm_oae_slices, oae_window, oae_step, fs=fs, trailing_silence=initial_silence+win_dur) # use the steady-state response for the fft (omit silence and onset)
        oae[li,:,:] = vbm_oae[:,:oae_size,:] / cochlea_scaling
    oae_fft, oae_nfft = compute_oae(oae, cf_no=oae_cf_no) # compute the fft of the oae response

# Rearrange the outputs, omit context and scale back to the original values
stim = stim[:,context_left:-context_right,:] # remove context from stim
vbm = vbm[:,context_left:-context_right,:] / cochlea_scaling # omit context from the uncropped outputs
vihc = vihc[:,context_left:-context_right,:] / ihc_scaling
ranf_hsr = ranf[0] / an_scaling
ranf_msr = ranf[1] / an_scaling
ranf_lsr = ranf[2] / an_scaling
del ranf

# Simulate CN and IC stages
if population_sim:
    print('Simulating IC-CN stages (Verhulstetal2018 v1.2)')
    # the CN/IC stage of the Verhulstetal2018 model (v1.2) is used
    cn = np.zeros(ranf_hsr.shape)
    an_summed = np.zeros(ranf_hsr.shape)
    ic = np.zeros(ranf_hsr.shape)
    for li in range(0,L.size):
        cn[li,:,:],an_summed[li,:,:]=nuclei.cochlearNuclei(ranf_hsr[li],ranf_msr[li],ranf_lsr[li],num_anfs[0],num_anfs[1],num_anfs[2],fs)
        ic[li,:,:]=nuclei.inferiorColliculus(cn[li,:,:],fs)
    # compute response waves 1, 3 and 5
    w1=nuclei.M1*np.sum(an_summed,axis=2)
    w3=nuclei.M3*np.sum(cn,axis=2)
    w5=nuclei.M5*np.sum(ic,axis=2)
    # EFR is the summation of the W1 W3 and W5 responses
    EFR = w1 + w3 + w5
    
    # EFR spectrum
    #EFR_sig = EFR[:,int((initial_silence+win_dur)*fs):int((initial_silence+stim_dur)*fs)] # keep only the signal part
    #nfft = next_power_of_2(EFR_sig.shape[1]) # size of fft
    #EFR_fft = np.fft.fft(EFR_sig,n=nfft) / EFR_sig.shape[1] # compute the fft over the signal part and divide by the length of the signal
    #nfft = int(nfft/2+1) # keep one side of the fft
    #EFR_fft_mag = 2*np.absolute(EFR_fft[:,:nfft])
    #freq = np.linspace(0, fs/2, num = nfft)
    
#################### Plot the responses ######################################
t = np.arange(0., ranf_hsr.shape[1]/fs, 1./fs)
ranf_hsr_no = ranf_hsr[:,:,No].T
ranf_msr_no = ranf_msr[:,:,No].T
ranf_lsr_no = ranf_lsr[:,:,No].T

# v_bm and V_IHC results
vbm_rms = rms(vbm, axis=1).T
ihc_rms = np.mean(vihc, axis=1).T
vbm_no = vbm[:,:,No].T
vihc_no = vihc[:,:,No].T

if oae_sim:
    oae_no = oae[:,:,oae_cf_no].T
    oae_freq = np.linspace(0, fs/2, num = oae_nfft)
    
    plt.figure(1, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1),plt.plot(1000*t[:oae_size],oae_no[:,::-1]),plt.grid()
    plt.xlim(0,50),plt.ylabel('Ear Canal Pressure [Pa]'),plt.xlabel('Time [ms]')
    plt.title('CF of ' + '%.2f' % CF[oae_cf_no] + ' Hz')
    plt.subplot(2,1,2),plt.plot(oae_freq/1000,20*np.log10(oae_fft.T/p0)),plt.grid()
    plt.ylabel('EC Magnitude [dB re p0]'),plt.xlabel('Frequency [kHz]'),plt.xlim(0,10)
    plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
    plt.tight_layout()

plt.figure(2, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
plt.subplot(2,2,1),plt.plot(1000*t,1e6*vbm_no[:,::-1]),plt.grid()
plt.xlim(0,50),plt.ylabel('$v_{bm}$ [${\mu}m$/s]'),plt.xlabel('Time [ms]')
plt.title('CF of ' + '%.2f' % CF[No] + ' Hz')
plt.subplot(2,2,2),plt.plot(CF/1000,20*np.log10(1e6*vbm_rms[:,::-1])),plt.grid()
plt.ylabel('rms of $v_{bm}$ [dB re 1 ${\mu}m$/s]'),plt.xlabel('CF [kHz]'),plt.xlim(0,8)
plt.title('Excitation Pattern')
plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
plt.subplot(2,2,3),plt.plot(1000*t,1e3*vihc_no[:,::-1]),plt.grid()
plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('$V_{ihc}$ [mV]')
plt.subplot(2,2,4),plt.plot(CF/1000,1e3*ihc_rms[:,::-1]),plt.grid()
plt.xlabel('CF [kHz]'),plt.ylabel('rms of $V_{ihc}$ [mV]'),plt.xlim(0,8)
plt.tight_layout()

# single-unit responses
plt.figure(3, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
plt.subplot(3,2,1),plt.plot(1000*t,ranf_hsr_no[:,::-1]),plt.grid()
plt.title('CF of ' + '%.2f' % CF[No] + ' Hz')
plt.xlim(0,100),plt.xlabel('Time [ms]'),plt.ylabel('HSR fiber [spikes/s]')
plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
plt.subplot(3,2,3),plt.plot(1000*t,ranf_msr_no[:,::-1]),plt.grid()
plt.xlim(0,100),plt.xlabel('Time [ms]'),plt.ylabel('MSR fiber [spikes/s]')
plt.subplot(3,2,5),plt.plot(1000*t,ranf_lsr_no[:,::-1]),plt.grid()
plt.xlim(0,100),plt.xlabel('Time [ms]'),plt.ylabel('LSR fiber [spikes/s]')
plt.tight_layout()

if population_sim:
    an_summed_no = an_summed[:,:,No].T
    cn_no = cn[:,:,No].T
    ic_no = ic[:,:,No].T
    
    # single-unit responses
    plt.subplot(3,2,2),plt.plot(1000*t,an_summed_no[:,::-1]),plt.grid()
    plt.title('CF of ' + '%.2f' % CF[No] + ' Hz')
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('sum AN [spikes/s]')
    plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
    # Spikes summed across all fibers @ 1 CF
    plt.subplot(3,2,4),plt.plot(1000*t,cn_no[:,::-1]),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('CN [spikes/s]')
    plt.subplot(3,2,6),plt.plot(1000*t,ic_no[:,::-1]),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('IC [spikes/s]')
    plt.tight_layout()

    # population responses
    plt.figure(4, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
    plt.subplot(4,1,1),plt.plot(1000*t,1e6*w1[::-1].T),plt.grid()
    plt.title('Population Responses summed across simulated CFs')
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('W-1 [${\mu}V$]')
    #plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
    plt.subplot(4,1,2),plt.plot(1000*t,1e6*w3[::-1].T),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('W-3 [${\mu}V$]')
    plt.subplot(4,1,3),plt.plot(1000*t,1e6*w5[::-1].T),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('W-5 [${\mu}V$]')
    plt.subplot(4,1,4),plt.plot(1000*t,1e6*EFR[::-1].T),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('EFR [${\mu}V$]')
    plt.tight_layout()
    
    # EFR spectrum
    #plt.figure(5, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
    #plt.plot(freq,EFR_fft_mag.T*1e6),plt.grid()
    #plt.title('EFR frequency spectrum')
    #plt.xlim(0,10000),plt.xlabel('Frequency [Hz]'),plt.ylabel('EFR Magnitude [${\mu}V$]')
    #plt.tight_layout()

plt.show()
