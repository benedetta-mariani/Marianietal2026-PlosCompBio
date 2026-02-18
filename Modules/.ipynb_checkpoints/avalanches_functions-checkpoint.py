"""
Avalanche Analysis Module

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from statsmodels.regression import linear_model as sm
from scipy.signal import find_peaks
import powerlaw as pwl

# ===========================================================
# Z-score normalization
# ===========================================================
def zscore(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std

# ===========================================================
# Threshold events per channel
# One event per suprathreshold excursion
# ===========================================================
def threshold_data(sample1, means, stds, thres):
    """
    One-event-per-negative-excursion thresholding.
    """
    sample1 = np.asarray(sample1)
    sample2 = np.zeros_like(sample1, dtype=int)

    for s in range(sample1.shape[1]):
        if stds[s] <= 0:
            continue

        sig = sample1[:, s]
        mask = (sig - means[s]) <= -thres * stds[s]

        diff = np.diff(mask.astype(int))
        starts = np.where(diff > 0)[0] + 1
        ends = np.where(diff < 0)[0] + 1

        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            ends = np.append(ends, len(sig))

        for a, b in zip(starts, ends):
            idx = a + np.argmin(sig[a:b])
            sample2[idx, s] = 1

    return sample2


def threshold_data_2(sample1, means, stds, thres):
    """
    Stricter negative excursion detection using sign change constraint.
    """
    sample1 = np.asarray(sample1)
    sample2 = np.zeros_like(sample1, dtype=int)

    for s in range(sample1.shape[1]):
        if stds[s] <= 0:
            continue

        sig = sample1[:, s]
        mask = (sig - means[s]) <= -thres * stds[s]
        sign_change = np.diff(np.sign(sig - means[s]))

        valid_cross = np.where(sign_change < 0)[0] + 1
        diff = np.diff(mask.astype(int))

        starts = np.where(diff > 0)[0] + 1
        ends = np.where(diff < 0)[0] + 1

        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            ends = np.append(ends, len(sig))

        for a, b in zip(starts, ends):
            if np.any((valid_cross >= a) & (valid_cross < b)):
                idx = a + np.argmin(sig[a:b])
                sample2[idx, s] = 1

    return sample2


def findpeaks(sig, thres, choose="neg", dist=200):
    """
    Peak detection using scipy.
    """
    sig2 = np.zeros_like(sig, dtype=int)
    sig = sig - np.mean(sig)

    if choose in ["neg", "posneg"]:
        p_neg = find_peaks(-sig, height=thres, distance=dist)[0]
        sig2[p_neg] = 1

    if choose == "posneg":
        p_pos = find_peaks(sig, height=thres, distance=dist)[0]
        sig2[p_pos] = 1

    return sig2


# ===========================================================
# Events per frame 
# ===========================================================
def events(sample):
    if sample.ndim > 2:
        sample = sample.reshape(sample.shape[0], -1)
    if sample.shape[1] > sample.shape[0]:
        raise Exception('Array must be time x channels')
    return np.sum(sample, axis=1).astype(int)

# ===========================================================
# Average inter-event interval
# ===========================================================
def avinterv(n):
    if n.ndim > 1:
        raise Exception('Array must be 1D')
    idx = np.where(n > 0)[0]
    intertempi = idx[1:] - idx[:-1]
    return int(round(np.mean(intertempi)))

# ===========================================================
# Bin activity
# ===========================================================
def binning(n, interv):
    pad = (-len(n)) % interv
    if pad > 0:
        n = np.pad(n, (0,pad))
    n_binned = n.reshape(-1, interv)
    n_sum = n_binned.sum(axis=1)
    active = (n_sum > 0).astype(int)
    start = np.flatnonzero(np.diff(active) > 0) + 1
    end = np.flatnonzero(np.diff(active) < 0) + 1
    if active[0] == 1:
        start = np.insert(start, 0, 0)
    if active[-1] == 1:
        end = np.append(end, len(active))
    sizes, durations = [], []
    for s,e in zip(start,end):
        sizes.append(n_sum[s:e].sum())
        durations.append(e-s)
    return sizes, durations
    
def bin_activity(activity, bin_size):
    """
    Temporal binning of activity.

    Parameters
    ----------
    activity : array (T,)
    bin_size : int

    Returns
    -------
    binned : array
    """
    pad = (-len(activity)) % bin_size
    if pad > 0:
        activity = np.pad(activity, (0, pad))

    return activity.reshape(-1, bin_size).sum(axis=1)
# ===========================================================
# Inter-avalanche times
# ===========================================================
def intertimes(data, interv, dt=1):
    ev = np.sum(data.T, axis=0).astype(bool)
    pad = (-len(ev)) % interv
    if pad > 0:
        ev = np.pad(ev, (0,pad))
    ev_binned = ev.reshape(-1, interv).sum(axis=1).astype(bool)
    padded = np.pad(ev_binned, (1,1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions==1)
    ends = np.flatnonzero(transitions==-1)
    silent = starts[1:] - ends[:-1] if len(starts) > 1 else np.array([])
    return silent * dt

# ===========================================================
# Average size for each duration
# ===========================================================
def sgivent(sizes, durations):
    """
    Fully vectorized conditional mean size.
    """
    sizes = np.asarray(sizes)
    durations = np.asarray(durations)

    unique_d, inverse = np.unique(durations, return_inverse=True)

    sums = np.bincount(inverse, weights=sizes)
    counts = np.bincount(inverse)

    means = sums / counts

    # std error
    var = np.bincount(inverse, weights=(sizes - means[inverse])**2) / counts
    errs = np.sqrt(var) / np.sqrt(counts)

    return unique_d, means, errs

# ===========================================================
# Delta from scaling law
# ===========================================================
def delta(alpha, salpha, tau, stau):
    val = (alpha-1)/(tau-1)
    err = np.sqrt((salpha/(tau-1))**2 + ((1-alpha)/(tau-1)**2*stau)**2)
    return val, err

# ===========================================================
# Raster plot
# ===========================================================
def Raster_Plot(sample, av, ax='default', color='red', alpha=0.3):
    if ax == 'default':
        fig,ax =plt.subplots(1,1)
    sample = sample.reshape(sample.shape[0], -1)
    binned = bin_activity(events(sample), av)
    times = np.arange(len(binned))*av
    s = np.array(binned)>0
    for i,t in enumerate(times):
        if s[i]:
            plt.fill_betweenx(np.arange(sample.shape[1]), t, t+av, color=color, alpha=alpha)
    for j in range(sample.shape[1]):
        idx = np.flatnonzero(sample[:,j]>0)
        plt.plot(idx, [j]*len(idx), '|', color='black', markersize=1.5)
    ax.set_ylabel('Unit')
    if ax =='default':
        return ax
def simple_raster_plot(raster):
    """
    Plot event raster.

    Parameters
    ----------
    raster : binary array (T, N)
    """
    t, ch = np.where(raster == 1)
    plt.figure()
    plt.scatter(t, ch, s=5)
    plt.xlabel("Time")
    plt.ylabel("Channel")
    plt.title("Raster Plot")
    plt.tight_layout()
    plt.show()

# ===========================================================
# Avalanche finder 
# ===========================================================
def avalanche_finder(S_shape_, coef=1):
    """
    Detects avalanches in a 1D time series of events.

    Parameters
    ----------
    S_shape_ : 1D array
        Time series of event counts per timestep (e.g., summed binary raster across channels).
    coef : float, default=1
        Scaling factor for the mean inter-spike interval used for binning.

    Returns
    -------
    sizes : 1D array
        Total number of events in each detected avalanche.
    durations : 1D array
        Duration (in bins) of each avalanche.
    S_shape_binned : 1D array
        Binned version of the input time series used for avalanche detection.
    shape_mean : list of lists
        For each unique duration, a list of the corresponding avalanche shapes (event sequences).
    freq : list
        Number of avalanches for each unique duration.
    """
    where_spikes = np.flatnonzero(S_shape_)
    if len(where_spikes)==0:
        return [],[],[],[],[]
    interspike_time = np.diff(where_spikes)
    mean_isi = int(round(np.mean(interspike_time)*coef))
    n = len(S_shape_)
    pad = (-n) % mean_isi
    S_shape_padded = np.pad(S_shape_, (0,pad))
    S_shape_binned = S_shape_padded.reshape(-1, mean_isi).sum(axis=1)
    padded = np.pad(S_shape_binned, (1,1))
    transitions = np.diff((padded>0).astype(int))
    t_in = np.flatnonzero(transitions==1)
    t_fin = np.flatnonzero(transitions==-1)
    durations = t_fin - t_in
    csum = np.concatenate(([0], np.cumsum(S_shape_binned)))
    sizes = csum[t_fin] - csum[t_in]
    # Compute shapes per duration
    unique_dur = np.unique(durations)
    shape_mean = [[] for _ in unique_dur]
    freq = []
    for i,d in enumerate(unique_dur):
        idx = np.flatnonzero(durations==d)
        freq.append(len(idx))
        for j in idx:
            shape_mean[i].append(S_shape_binned[t_in[j]:t_fin[j]])
    return sizes, durations, S_shape_binned, shape_mean, freq

# ===========================================================
# Power-law fitting
# ===========================================================
def exponent(sample, maxxmin="default", xmax="default", lim=4):
    if maxxmin=="default": maxxmin = max(sample)
    if xmax=="default": xmax = max(sample)
    fit = pwl.Fit(sample, xmin=(1,maxxmin+1), xmax=xmax, parameter_range={'alpha':[1,lim]}, discrete=True)
    return fit.power_law.alpha, fit.power_law.sigma

def return_param(sample, maxxmin="default", xmax="default", lim=4):
    if maxxmin=="default": maxxmin = max(sample)
    if xmax=="default": xmax = max(sample)
    fit = pwl.Fit(sample, xmin=(1,maxxmin+1), xmax=xmax, parameter_range={'alpha':[1,lim]}, discrete=True)
    return fit.power_law.xmin, fit.power_law.alpha, fit.power_law.sigma

# ===========================================================
# Scaling ⟨S⟩(T)
# ===========================================================
def scaling_new(sizes, durations, avinterval,
                lim1 = 4, lim2 = 4,
                tau = "default", errtau = "default",
                alpha = "default", erralpha = "default",
                maxxminsizes = "default", maxxmindur = "default",
                xmaxsizes = "default", xmaxdur = "default"):
            
    if maxxminsizes ==  "default":
        maxxminsizes = max(sizes)
    if xmaxsizes ==  "default":
        xmaxsizes = max(sizes)
    if maxxmindur ==  "default":
        maxxmindur = max(durations)
    if xmaxdur ==  "default":
        xmaxdur = max(durations)
    
    if tau == "default" and errtau == "default":
        xmin,tau,errtau = return_param(sizes,maxxmin = maxxminsizes,xmax =xmaxsizes,lim = lim1)
        
    if alpha == "default" and erralpha == "default":
        xmin,alpha,erralpha = return_param(durations,maxxmin = maxxmindur,xmax =xmaxdur,lim = lim2)

    pred =  delta(alpha, erralpha, tau, errtau)[0]
    errpred = delta(alpha, erralpha,tau, errtau)[1]
    
    durations = np.array(durations)*avinterval
    xmin1 = min(sizes)
    xmin2 = min(durations)
    
    prova = np.array([np.asarray(sizes), np.asarray(durations)])
    prova = prova.transpose()
    prova2 = [0 for i in range(len(prova))]
    
    for r in range(len(prova)):
        if prova[r][0] < xmin1 or prova[r][1] < xmin2 :
            prova2[r] = False
        else:
            prova2[r] = True
  
    new = prova[prova2]
    #print(len(new[:,0]), len(new[:,1]))
    a, b, c = sgivent(new[:,0],new[:,1])
    x = np.hstack((np.log10(a).reshape(-1,1), np.ones(len(a)).reshape(-1,1)))
    
    y = np.log10(b).reshape(-1,1)

    ols = sm.OLS(y,x)
    ols_result = ols.fit()
    fit = ols_result.params[0]
    errfit = ols_result.bse[0]
    inter =ols_result.params[1]    
    print('Prediction from crackling noise relation: delta = ',pred, '+-', errpred)
    print('Fit from of average size given duration points: delta = ',fit, '+-', errfit)
    
    x = np.arange(min(durations),max(durations))
    
    return a, b, c, x, inter, pred, errpred, fit, errfit

# ===========================================================
# Plotting crackling noise
# ===========================================================
def plot_crackling_noise(ax, args, cpoints, cpred, cfit,
                         xmin=1, xmax=1000, alphamarker=0.5,
                         s=5, lwline=3, label='', prediz=True, fontsize=15):
    """
    Plot crackling noise: avalanche sizes vs durations with scaling law fits.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on.
    args : tuple/list
        Tuple containing (sizes, durations, a, b, c, x, inter, pred, errpred, fit, errfit)
    cpoints, cpred, cfit : color
        Colors for points, predicted fit, and actual fit.
    xmin, xmax : float
        Range of durations to display.
    alphamarker : float
        Transparency for scatter points.
    s : float
        Scatter marker size.
    lwline : float
        Line width for fits.
    label : str
        Label used in legend for predicted fit.
    prediz : bool
        Whether to plot predicted scaling.
    fontsize : int
        Font size for axis labels.

    Returns
    -------
    None
    """
    sizes, durations, a, b, c, x, inter, pred, errpred, fit, errfit = args

    x = np.linspace(xmin, xmax, 1000)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Scatter mean sizes
    ax.scatter(a[(a > xmin) & (a < xmax)], b[(a > xmin) & (a < xmax)],
               color=cfit, s=s, alpha=alphamarker)

    # Predicted scaling
    if prediz:
        ax.plot(x, (10**inter) * x**pred, cpred,
                label=r'$\delta^\mathrm{' + label +
                      r'}_\mathrm{pred} =%2.2f \pm %2.2f$' %
                      (round(pred, 2), round(errpred + 0.01, 2)),
                lw=lwline, ls='--', zorder=10000)

    # Fit scaling
    ax.plot(x, (10**inter) * x**fit, color=cfit,
            label=r'$\delta^\mathrm{' + label +
                  r'}_\mathrm{fit} \,\,=%2.2f \pm %2.2f$' %
                  (round(fit, 2), round(errfit + 0.01, 2)),
            lw=lwline, ls='--', zorder=10000)

    # Confidence interval for fit
    ax.fill_between(x, (10**inter) * x**(fit - 3*errfit),
                    (10**inter) * x**(fit + 3*errfit),
                    color=cfit, lw=0, alpha=0.1)

    # Confidence interval for prediction
    if prediz:
        ax.fill_between(x, (10**inter) * x**(pred - 3*errpred),
                        (10**inter) * x**(pred + 3*errpred),
                        color=cpred, lw=0, alpha=0.1)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'Durations [\# of time bins]', fontsize=fontsize, labelpad=10)
    ax.set_ylabel(r'Sizes [\# of events]', fontsize=fontsize, labelpad=10)


def returnbin(n,interv): 
    v = [] 
    if len(n)%interv > 0: 
        add = (int(len(n)/interv) + 1)* interv - len(n) 
        n = n.tolist() 
        for i in range(add): 
            n = n + [0] 
    n= np.asarray(n).reshape(int(len(n)/interv), interv) 
    for z in range(len(n)): 
        if np.any(n[z]): 
            v.append(1) 
        else: 
            v.append(0) 
    return v,n


