import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd
import glob
from pandas.core.algorithms import isin
from Tools.utils import auto_tab

evalloss_files = './Files/tfrecords_10_folds/evalloss/*.csv'
trainloss_files = './Files/tfrecords_10_folds/trainloss/*.csv'
evalloss_files, trainloss_files = auto_tab(evalloss_files, trainloss_files)

def read_csv(file):
    assert os.path.exists(file)
    df = pd.read_csv(file)
    data = df['Value'].to_numpy()
    return data

def get_colors(num):
    rgb = np.random.rand(num, 3)
    return rgb

def get_foldid(file):
    key = os.path.basename(file).split('_')[2]
    if key == '00':
        key = 1
    else:
        key = int(key)
    return key

def load_csvfile(files:list):
    data = {}
    for fold_id, f in enumerate(files):
        # fold_id = get_foldid(f)
        data[fold_id] = read_csv(f)
    return data

def plot_data_st1(data:dict, name, type, drop=6):
    """
        plot data : strategy 1. plot all loss in one figure. 
    """
    max_step = np.max([len(d) for d in data.values()])
    fig, axs = plt.subplots()
    colors = get_colors(10)
    max_pos = list(range(max_step-drop))
    for id in range(len(data)):
        records = data[id]
        pos = range(len(records)-drop)
        axs.plot(pos, records[:-drop] if drop>0 else records, label=f'fold {id}')
        # axs.plot(pos[-1], 0, '^', color=plt.gca().lines[-1].get_color())  # plot on the step axis.
    axs.set_xticks(max_pos)
    axs.set_xlabel('Step')
    axs.set_ylabel(f'{type}Loss')
    axs.legend()

    plt.savefig(f'./Visual/tfrecords_plot/{name}', dpi=96, transparent=False, bbox_inches='tight')
    plt.close('all')

    
def main():
    # plot the train loss. 
    data = load_csvfile(trainloss_files)
    plot_data_st1(data, 'trainLosses', 'train')
    data = load_csvfile(evalloss_files)
    plot_data_st1(data, 'evallosses', 'eval')

    # data = load_csvfile(trainloss_files)
    # plot_data_st1(data, 'trainLosses_none_drop', 'train', drop=1)
    # data = load_csvfile(evalloss_files)
    # plot_data_st1(data, 'evallosses_none_drop', 'eval', drop=1)
    

if __name__ == '__main__':
    main()
    