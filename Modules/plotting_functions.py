import powerlaw as pwl
import sys
#from avalanches_functions import *
sys.path.append("Modules")
#C:/Users/Benedetta/Desktop/Criticality in barrel cortex")
from powerlaw_fit import *
from stats import *
from power import *

# ===========================================================
# Return power-law fit parameters
# ===========================================================
def return_param(sample, maxxmin="default", xmax="default", lim=4):
    """
    Fit a discrete power-law to the data and return xmin, alpha, sigma.

    Parameters
    ----------
    sample : array-like
        Data to fit.
    maxxmin : int or "default"
        Maximum xmin cutoff for fitting.
    xmax : int or "default"
        Maximum value considered for fitting.
    lim : float
        Maximum value for alpha in the fit.

    Returns
    -------
    xmin : float
        Minimum cutoff of the power-law.
    alpha : float
        Exponent of the power-law.
    sigma : float
        Standard error of alpha.
    """
    if maxxmin == "default":
        maxxmin = max(sample)
    if xmax == "default":
        xmax = max(sample)
  
    ypred = pwl.Fit(sample, xmin=(1, maxxmin+1), xmax=xmax, parameter_range={'alpha':[1., lim]}, discrete=True)
    return ypred.power_law.xmin, ypred.power_law.alpha, ypred.power_law.sigma


# ===========================================================
# Plot avalanche statistics (PDF)
# ===========================================================
def plot_av_statistics(ax, args, color, xmin, xmax, label=None,
                       marker='o', s=100, lw=1.5, 
                       lwline=3, alphaline=0.5,
                       plot_fit=True, expo=False):
    """
    Plot empirical PDF of avalanche statistics and optional power-law fit.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot on.
    args : tuple
        (centredbin, pdfnorm, x, px_fit, exp, errexp)
    color : str
        Color for plotting.
    xmin, xmax : float
        Plot limits.
    label : str
        Label for fit line.
    marker, s, lw : marker style parameters
    plot_fit : bool
        Whether to plot the theoretical fit line.
    expo : bool
        Whether to overlay an exponential fit (from pwl.Fit).
    """
    centredbin, pdfnorm, x, px_fit, exp, errexp = args
    centredbin = np.array(centredbin)
    pdfnorm = np.array(pdfnorm)
    
    ax.scatter(centredbin[centredbin < xmax], pdfnorm[centredbin < xmax],
               color=color, marker=marker, s=s, lw=lw, facecolors='none')
    ax.plot(centredbin, pdfnorm, lw=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
        
    if plot_fit:
        x_fit = np.linspace(xmin, xmax, 1000)
        ax.plot(x_fit, .5 * x_fit**-exp,
                ls='--', color=color, lw=lwline, alpha=alphaline, label=label)
        
    if expo:
        px_fit.exponential.plot_pdf(ls='--', color=color, lw=lwline, alpha=alphaline, ax=ax)


# ===========================================================
# LogScript_new: compute binned PDF and fit parameters
# ===========================================================
def LogScript_new(x_data, av, lim, maxxmin, xmax='default', expo=False, nbins="default"):
    """
    Process avalanche data: select data above xmin, compute PDF,
    optionally fit power-law and exponential distributions.

    Parameters
    ----------
    x_data : array-like
        Avalanche sizes or durations.
    av : float
        Scaling factor (typically 1 or bin width).
    lim : float
        Maximum alpha for power-law fitting.
    maxxmin, xmax : int or 'default'
        Cutoffs for fitting.
    expo : bool
        Whether to include exponential fit.
    nbins : int or 'default'
        Number of histogram bins.

    Returns
    -------
    centredbin : array
        Bin centers of PDF.
    pdfnorm : array
        Normalized PDF.
    x : array
        Unique values from x_data >= xmin.
    px_fit : object
        Fitted distribution (power-law or exponential).
    exp : float
        Estimated exponent alpha.
    errexp : float
        Standard error of exponent.
    xmin : float
        Minimum cutoff used for fitting.
    """
    if maxxmin == "default":
        xmin, exp, errexp = return_param(x_data, max(x_data), max(x_data), lim)
    else:
        xmin, exp, errexp = return_param(x_data, maxxmin, max(x_data), lim)
    
    if xmax == 'default':
        xmax = max(x_data)
    
    new1 = [val for val in x_data if val >= xmin]
    
    if nbins == "default":
        nbins = len(pwl.pdf(x_data)[0])
        
    print('Nbins estimated are are', len(pwl.pdf(x_data)[0]))
    
    nrep_synth = 0
    x, nx = xdata_to_xnx(new1, norm=False, xmin=xmin, xmax=xmax)
    result = fit_power_disc_sign(x, nx, xmin=xmin, xmax=xmax, nrep_synth=nrep_synth) 
    Alpha = result['alpha']
    print('Alpha is', Alpha)
    new = np.array(new1)
    
    bins = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), num=nbins, endpoint=True)
    
    if av != 1:
        new *= av
        xmin *= av
        xmax *= av
        x_data2 = np.array(x_data) * av
        x, nx = xdata_to_xnx(new, norm=False, xmin=xmin, xmax=xmax)
        bins *= av
    else:
        x_data2 = x_data
        
    px_fit = pdf_power_disc(x, xmin, xmax, Alpha)
    
    pdf = [[] for _ in range(len(bins)-1)]
    for i in range(1, len(bins)):
        for val in x_data2:
            if bins[i-1] <= val < bins[i]:
                pdf[i-1].append(val)

    pdfnorm = [len(pdf[i-1])/((bins[i]-bins[i-1])*len(x_data2)) for i in range(1,len(bins))]  
    centredbin = [(bins[i]+ bins[i+1])/2 for i in range(len(bins)-1)]
    
    if expo:
        fitexp = pwl.Fit(x_data, xmin=min(x_data), xmax=max(x_data), discrete=True)
        return centredbin, pdfnorm, x, fitexp, exp, errexp, xmin
    
    return centredbin, pdfnorm, x, px_fit, exp, errexp, xmin