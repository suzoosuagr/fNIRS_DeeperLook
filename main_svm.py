# %% initialization
import numpy as np
from numpy.core.fromnumeric import shape
from Data.Dataset.fnirs import fNIRS_mb_label_balance_leave_subject
import pandas as pd
from Experiments.Config.issue03 import *
from Tools.logger import *
import os
import json
import scipy.stats as stats
from skfeature.function.similarity_based import fisher_score as Fs
from Tools.utils import ensure
from tqdm import trange

args = EXP01('debug', 'standard.log')
args.logpath = os.path.join(args.log_root, args.name, args.logfile)
setup_global_logger(args.mode, logging.INFO, logpath=args.logpath)
save_root = './Files/svm/'
ensure(save_root)

# %% features inside time window 
def mean(data):
    """
    @parames:
        - data: ndarray with shape [N,C,Channels]; contining oxy or deoxy only. 
                N is the time. calculate mean over time. 
    @outputs:
        - out: shape [1,C,Channels]
    """
    m = np.mean(data, axis=0, keepdims=False)
    return m

def var(data):
    v = np.var(data, axis=0, keepdims=False)
    return v

def skew(data):
    sk = stats.skew(data, axis=0)
    return sk

def kurtosis(data):
    k = stats.kurtosis(data, axis=0)
    return k

def slope(data):
    dy = data[-1,:,:] - data[0,:,:]
    dx = len(data)
    slop = dy/dx
    return slop

# %%
def feature_selection(dataset, featFunc, topk=6):
    feat_pool = []
    label_pool = []
    selected_feat = []
    for i in trange(len(dataset)):
        data, label, oxy_file = dataset[i]
        feat = featFunc(data)
        feat_pool.append(feat)
        label_pool.append(label)
    feat_pool = np.stack(feat_pool, axis=1) # [C, N, Channels], where C is the oxy and deoxy stuff
    for i in trange(2):
        feat_ = feat_pool[i,:,:]
        fisher_score = Fs.fisher_score(feat_, label_pool)
        rank = Fs.feature_ranking(fisher_score)
        selected_feat.append(rank[:topk])
    return selected_feat, feat_pool, label_pool

def kfoldFeatSave(args, k, featFunc):
    Basic_Name = args.name
    feat_order = ['Oxy', 'Deoxy']
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping = json.load(jsf)
    for fold_id in range(k):
        args.data_config['train_ids'] = fold_id_mapping[str(fold_id)]['train_ids']
        args.data_config['eval_ids'] = fold_id_mapping[str(fold_id)]['eval_ids']
        args.name = "{}_{:02}".format(Basic_Name, fold_id)
        print(f"Runing {args.name} | eval on {args.data_config['eval_ids']}")

        train_dataset = fNIRS_mb_label_balance_leave_subject(\
                    list_root = args.list_path,
                    steps = args.steps_sizes,  
                    mode='train',
                    data_config=args.data_config,
                    runtime=True,
                    fold_id=fold_id)
        test_dataset = fNIRS_mb_label_balance_leave_subject(\
                list_root = args.list_path,
                steps = args.steps_sizes,
                mode='eval',
                data_config=args.data_config,
                runtime=True,
                fold_id=fold_id)

        print(f"Train : {len(train_dataset)}")
        print(f"Test : {len(test_dataset)}")
        saveKroot = os.path.join(save_root, '{:02}'.format(fold_id))
        ensure(saveKroot)
        fswriter = open(os.path.join(saveKroot, f'featSelection_{featFunc.__name__}.txt'), 'w')
        fs, feat_pool, label_pool = feature_selection(train_dataset, featFunc)
        np.save(os.path.join(saveKroot, f'featPool_{featFunc.__name__}.npy'), feat_pool)
        np.save(os.path.join(saveKroot, f'labelPool_{featFunc.__name__}.npy'), label_pool)
        for i in range(2):
            fswriter.write(f'Selected {feat_order[i]}=='+','.join([str(s) for s in fs[i].tolist()])+'\n')
        fswriter.close()

def load_data(root, fold_id):
    feat_path = os.path.join(root, fold_id, 'featPool.npy')
    label_path = os.path.join(root, fold_id, 'labelPool.npy')
    fs_path = os.path.join(root, fold_id, 'featSelection.txt')
    featPool = np.load(feat_path)
    labelPool = np.load(label_path)
    with open(fs_path, 'r') as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]
        oxy_index = lines[0].split('==')[-1].split(',')
        deoxy_index = lines[1].split('==')[-1].split(',')
        oxy_index = [int(v) for v in oxy_index]
        deoxy_index = [int(v) for v in deoxy_index]

    return featPool, labelPool, oxy_index, deoxy_index

if __name__ == "__main__":
    for func in [mean, var, skew, kurtosis, slope]:
        kfoldFeatSave(args, k=10, featFunc=func)
        

    # featPool, LabelPool, oxy_index, deoxy_index = load_data(save_root, '00')
    # print("Read. ")
# %%
