from scipy.signal import find_peaks
import numpy as np
import sys
sys.path.append("Modules")

import matplotlib.pyplot as plt
from avalanches_functions import *
def findpeaks(sig, thres, choose, dist = 10):
    """
    Finds peaks in a time series.
    Set eventual other constraints in the scipy.signal function find_peaks
    """
    sig2 = np.zeros(sig.shape, dtype = int)
    if choose == "neg":
        p1 = find_peaks(-(sig - np.mean(sig)), height = thres, distance = dist)[0]
        if np.any(p1):
            sig2[p1] = 1
            
    if choose == "posneg":
        p1 = find_peaks(-(sig - np.mean(sig)), height = thres)[0]
        p2 = find_peaks((sig - np.mean(sig)), height = thres)[0]
        if np.any(p1.tolist() + p2.tolist()):
            sig2[np.array(p1.tolist()+ p2.tolist())] = 1
    return sig2

def firingrate(array, timebin=250):
    # array shape: trials x time x channels
    time = array.shape[1]
    rest = int(timebin)*int(time/timebin + 1) - int(time) 

    firingrates = np.zeros((array.shape[0],int((time + rest)/timebin), timebin, array.shape[2]))

    for l in range(len(array)):
        firingrates[l] =np.append(array[l,:], np.zeros((rest, array.shape[2]))).reshape(int((time+rest)/timebin), timebin,array.shape[2])

    firingrates = np.sum(firingrates, axis = 2)/timebin
    return firingrates
    
def Thres(coef,x):
    """
    Calculates Quiroga detection threshold
    """
    return coef*np.median(np.abs(x)/0.6745)

def RasterPlot(spikes, conv, ax = None):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    idx = spikes > 0
    times = np.arange(0,len(spikes),1)
    #print(times.shape)
    for l in range(spikes.shape[1]):
        x = times[idx[:,l]]
        ax.scatter(x*conv, [l for i in range(len(x))], marker ='s', c = 'black',s = 2.)
        
def RasterPlot2(sample2, av, fs = 500,ax = None, av_color = 'gray',
               nsize = 1.5, alpha = 0.3):
    """
    Parameters
    --------
    sample : Array of discretized data. Dimensions : temporal dim x spatial dim1 (x spatial dim2)
    av : width of temporal bin 
    Returns
    --------
    Plots the Raster Plots and the detected avalanches (an avalanche is preceded and followed by white bins)
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    s,sample = returnbin(np.sum(sample2,1),av)

    if s[0] == 1:
        start_av = True
    else:
        start_av = False
    #print(np.where(np.diff(s) != 0)[0].shape)
    xtime = np.arange(0,len(sample2),1)/fs
    s = xtime[(np.where(np.diff(s) != 0)[0]+1)*av]
    #rint(s[-1])
    if start_av:
        s = np.concatenate([[0], s])
        
    #print(s.size)
    if s.size % 2 != 0:
        s = np.concatenate([s, [xtime[-1]]])
    for sval in s.reshape(-1, 2):
        ax.axvspan(sval[0], sval[1], 0.01, 0.99, color = av_color, alpha = alpha)
    for j in range(sample2.shape[1]):
        idx = np.where(sample2[:,j]> 0)[0]
        ax.plot(xtime[idx], np.ones(len(idx))*j, '|', color = 'black', markersize = nsize)

def compute_density(data, mean_interspike_time, coef, 
                   ):
    ev = np.array(np.sum(data, axis = 1), dtype = int)
    density = np.zeros(ev.shape)
    ev2 = np.array(np.array(ev, dtype = bool), dtype = int)
    T = mean_interspike_time*coef
    for g in range(len(ev)):
        density[g] = np.sum(ev2[g:g+T])/T
    return density

def compute_fr(data, mean_interspike_time, coef, ):
    ev = np.array(np.sum(data, axis = 1), dtype = int)
    density = np.zeros(ev.shape)
    T = mean_interspike_time*coef
    for g in range(len(ev)):
        density[g] = np.sum(ev[g:g+T])/T
    return density
    
def concatenat_spikes(array,idx1,idx2, rep,fs):
    g = array[0,:,fs*idx1:fs*idx2]
    for h in range(rep):
        g = np.concatenate((g, array[h,:,fs*idx1:fs*idx2]),axis = 1)
    return g.T
    