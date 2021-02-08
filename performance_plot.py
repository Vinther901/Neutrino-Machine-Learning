import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def performance_plot(pred,true,Energies,ylim=None):
    fig, ax = plt.subplots(figsize=(10,7))

    sort = np.argsort(Energies)

    Energies = Energies[sort]
    res = pred[sort] - true[sort]

    slices = np.arange(0,4.5,0.5)

    quantiles = np.zeros((len(slices) - 1,3))
    xes = np.zeros(len(slices) - 1)

    for i in range(1,len(slices)):
        mask = (Energies > slices[i-1])&(Energies < slices[i])
        quantiles[i-1] = np.quantile(res[mask],(0.25,0.5,0.75))
        xes[i-1] = np.mean(Energies[mask])
    
    ax.errorbar(x = xes, y = quantiles[:,1], yerr = abs(quantiles[:,1] - quantiles[:,[0,2]].T),fmt='none')
    ax.plot(xes,quantiles[:,1],'k.')
    ax.set(xlim=(-0.2,4.2),ylim=ylim)
    ax.hlines(0,0,4)
    plt.grid()
    fig.show()

    fig2, ax2 = plt.subplots(figsize=(10,7))
    for i in range(quantiles.shape[1]):
        ax2.plot(xes,quantiles[:,i])
    ax2.set(xlim=(-0.2,4.2),ylim=ylim)
    ax2.hlines(0,0,4)
    plt.grid()
    fig2.show()

    return quantiles