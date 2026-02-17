import powerlaw as pwl
import sys
from avalanches_functions import *
sys.path.append("C:/Users/Benedetta/Desktop/Criticality in barrel cortex")
from powerlaw_fit import *
from stats import *
from power import *
def return_param(sample,maxxmin = "default",xmax = "default",lim = 4):

    if maxxmin ==  "default":
        maxxmin = max(sample)
    if xmax ==  "default":
        xmax = max(sample)
  
    ypred = pwl.Fit(sample, xmin = (1,maxxmin+1),xmax = xmax, parameter_range = {'alpha' : [1.,lim]}, discrete = True)
    return ypred.power_law.xmin, ypred.power_law.alpha, ypred.power_law.sigma
def plot_av_statistics(ax, args, color, xmin, xmax, label = None,
                       marker = 'o', s = 100, lw = 1.5, 
                       lwline = 3, alphaline = 0.5,
                       plot_fit = True, expo = False):

    
    centredbin, pdfnorm, x, px_fit, exp, errexp = args
    centredbin = np.array(centredbin)
    pdfnorm = np.array(pdfnorm)
    
    ax.scatter(centredbin[centredbin < xmax], pdfnorm[centredbin < xmax],
               color = color, marker = marker, s = s, lw = lw,
               facecolors='none')
    ax.plot(centredbin, pdfnorm, lw = 0)
    ax.set_xscale('log')
    ax.set_yscale('log')
        
    if plot_fit:
        x = np.linspace(xmin, xmax, 1000)
        ax.plot(x, .5*x**-exp,
                ls = '--', color = color, lw = lwline, alpha = alphaline, label = label)
        
        
    if expo:
        px_fit.exponential.plot_pdf(ls = '--', color = color, lw = lwline, 
                                    alpha = alphaline, ax = ax)
def LogScript_new(x_data, av, lim, maxxmin, xmax= 'default',expo = False, nbins = "default"):
    if maxxmin == "default":
        xmin, exp, errexp = return_param(x_data,max(x_data),max(x_data),lim)
    else:
        xmin, exp, errexp = return_param(x_data,maxxmin,max(x_data),lim)
    if xmax == 'default':
        xmax = max(x_data)
    
    new1 = []
    for i in range(len(x_data)):
        if x_data[i] >= xmin:
            new1.append(x_data[i])
        
    if nbins == "default":
        nbins = len(pwl.pdf(x_data)[0])
        
    
    print('NBins for non random data are',len(pwl.pdf(x_data)[0]))
    
    nrep_synth = 0
    x,nx = xdata_to_xnx(new1,norm=False,xmin=xmin,xmax=xmax)
    result = fit_power_disc_sign(x,nx,xmin=xmin,xmax=xmax,nrep_synth=nrep_synth) 
    Alpha = result['alpha']
    print('Alpha is', Alpha)
    new = np.array(new1)
    
    bins = np.logspace(np.log10(min(x_data)),np.log10(max(x_data)), num = nbins, endpoint= True)
    
    if av!= 1:
        new = new*av
        xmin = xmin*av
        xmax = xmax*av
        x_data2 = np.array(x_data)*av
        x,nx = xdata_to_xnx(new,norm=False,xmin=xmin,xmax=xmax)
        bins = bins*av
    else:
        x_data2 = x_data
        
    px_fit = pdf_power_disc(x,xmin,xmax,Alpha)
    
    pdf = [[] for i in range(len(bins)-1)]
    for i in range(1,len(bins)):
        for l in range(len(x_data2)):
            if x_data2[l]>=bins[i-1] and x_data2[l] < bins[i]:
                pdf[i-1].append(x_data2[l])

    pdfnorm = [len(pdf[i-1])/((bins[i]-bins[i-1])*len(x_data2)) for i in range(1,len(bins))]  
    centredbin = [(bins[i]+ bins[i + 1])/2 for i in range(len(bins)-1)]
    if expo == False:
        return centredbin, pdfnorm, x, px_fit, exp, errexp, xmin
    if expo == True:
        fitexp = pwl.Fit(x_data, xmin = min(x_data), xmax = max(x_data), discrete = True)
        return centredbin, pdfnorm, x, fitexp, exp, errexp, xmin