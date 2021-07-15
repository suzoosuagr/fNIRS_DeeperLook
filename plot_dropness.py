# %% 
import scipy.stats as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 15,
          'figure.figsize': (8, 4),
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':15,
         'ytick.labelsize':18}
pylab.rcParams.update(params)

def text_parser(txtPath):
    assert os.path.exists(txtPath)
    wml_f1 = []
    vpl_f1 = []
    with open(txtPath, 'r') as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]

        for l in lines:
            w, p = l.split('\t')
            wml_f1.append(float(w))
            vpl_f1.append(float(p))
    return wml_f1, vpl_f1    

def plot_fig(wml_f1, vpl_f1, interval_wml, interval_vpl):
    fig, axs = plt.subplots()
    pos = list(range(len(wml_f1)))
    axs.plot(pos, wml_f1, '^', label='wml f1-score', color='b')
    axs.plot(pos, vpl_f1, 'o', label='vpl f1-score', color='g')

    axs.axhline(interval_wml[0], ls='--', color='b', alpha=0.3)
    axs.axhline(interval_wml[1], ls='--', color='b', alpha=0.3)

    axs.axhline(interval_vpl[0], ls='--', color='g', alpha=0.3)
    axs.axhline(interval_vpl[1], ls='--', color='g', alpha=0.3)

    axs.set_xticks(pos)
    axs.set_xlabel('Position Index')
    axs.set_ylabel('Performance Drop')
    axs.legend()

    plt.savefig('./Visual/perf_drop.png', dpi=80, transparent=False, bbox_inches='tight')
    plt.close('all')


pdropResults = './Files/performance_drop/4ch.txt'
wml_f1, vpl_f1 = text_parser(pdropResults)
# plot_fig(wml_f1, vpl_f1)
data = wml_f1
wmlInterval = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
# print(f"wml: low->{low} | high->{high}")
data = vpl_f1
vplInterval = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
# print(f"vpl: low->{low} | high->{high}")

plot_fig(wml_f1, vpl_f1, wmlInterval, vplInterval)

# %%

