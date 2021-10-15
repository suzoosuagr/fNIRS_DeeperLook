# %% initialization
from Tools.metric import PerformanceTestEnsembleMB
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.npyio import load
from Data.Dataset.fnirs import Ann_mb_label_balance
import pandas as pd
from Experiments.Config.issue03 import *
from Tools.logger import *
import torch.utils.data as data 
from Tools import env_init
import os
import json
import scipy.stats as stats
from skfeature.function.similarity_based import fisher_score as Fs
from Tools.utils import ensure
from tqdm import trange
from Tools.engine import Ann_Engine
import torch.optim as optim
import torch
import torch.nn as nn
from Model.models import ANN
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest="test", default="debug", type=str)
    parser.add_argument("-l", "--log", dest="logfile", default='debug.log', type=str)
    
    return parser.parse_args()

parser = parse_args()
args = EXP02('debug', 'standard.log')
warning("STARTING >>>>>> {} ".format(args.name))
args.logpath = os.path.join(args.log_root, args.name, args.logfile)
ngpu, device, writer = env_init(args, logging.INFO)
args.ngpu = ngpu

data_root = './Files/svmfisherZscoreWML/'
def update_loader(fold_id, args_, data_root):
    train_dataset = Ann_mb_label_balance(data_root, mode='train', funcList=['mean', 'var', 'skew', 'kurtosis', 'slope', 'peak'], fold_id=fold_id)
    eval_dataset  = Ann_mb_label_balance(data_root, mode='eval', funcList=['mean', 'var', 'skew', 'kurtosis', 'slope', 'peak'], fold_id=fold_id)
    test_dataset  = Ann_mb_label_balance(data_root, mode='test', funcList=['mean', 'var', 'skew', 'kurtosis', 'slope', 'peak'], fold_id=fold_id)


    train_loader = data.DataLoader(train_dataset, batch_size=args_.batch_size, shuffle=True, drop_last=args_.drop_last)
    eval_loader = data.DataLoader(eval_dataset, batch_size=args_.batch_size, shuffle=False, drop_last=args_.drop_last)
    test_loader = data.DataLoader(test_dataset, batch_size=args_.batch_size, shuffle=False, drop_last=args_.drop_last)

    info(f"Train:{len(train_dataset)}|Eval:{len(eval_dataset)}|Test:{len(test_dataset)}|")

    return train_loader, eval_loader, test_loader

def update_model(model_name):
    if model_name == 'ANN':
        model = ANN(12*52, [256, 128], 3)
    else:
        raise NotImplementedError
    return model

def run_summary():
    model = ANN(12*52, [256, 128], 3).to(device)
    psummary = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters {}".format(psummary))
       

def run_kfolds(args, basic_name, data_root, k=10, mode='test'):
    count = 0
    accu = 0
    Basic_Name = basic_name
    IDS = args.data_config["ids"].copy()
    ensemble_metrics = PerformanceTestEnsembleMB(size=3)
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping = json.load(jsf)
    for fold_id in range(k):
        args.name = "{}_{:02}".format(Basic_Name, fold_id)
        info(f"{fold_id_mapping[str(fold_id)]}")
        train_loader, eval_loader, test_loader = update_loader(fold_id, args, data_root)
        model = update_model(args.model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        exe = Ann_Engine(train_loader, eval_loader, test_loader, args, writer, device)
        if 'train' in mode or 'full' in mode:
            exe.train(model, optimizer, criterion, None)
        if 'test' in mode or 'full' in mode:
            if fold_id == k-1:
                last = True
            else:
                last = False
            ensemble_metric = exe.ensemble_test(model, optimizer, ensemble_metrics, Basic_Name, last=last)
    return ensemble_metrics.value()


if __name__ == "__main__":
    # grid search
    # basic_name = args.name
    # search_records = open('./Files/ann/search_results_ANN_a.txt', 'w')
    # for path in ['./Files/svmfisherZscoreWML/', './Files/svmfisherZscoreVPL/']:
    #     max_search_accu = 0
    #     for lr in [1e-3]:
    #         for batch_size in [32]:
    #             args.lr = lr
    #             args.batch_size = batch_size
    #             confMat, accu, precision, recall, f1 = run_kfolds(args, basic_name, data_root=path, k=10, mode='full')
    #             if accu > max_search_accu:
    #                 max_search_accu = accu
    #                 search_records.write("="*20+'\n')
    #                 search_records.write(f"MAX ACCU UPDATED: LR->{lr}, BatchSize->{batch_size}\n")
    #             search_records.write("+"*20+'\n')
    #             search_records.write("ACCURACY_{} = {}\n".format(path.split('/')[-2], accu))
    #             search_records.write("Precision{} = {}\n".format(path.split('/')[-2], precision))
    #             search_records.write("Recall_{} = {}\n".format(path.split('/')[-2], recall))
    #             search_records.write("F1_{} = {}\n".format(path.split('/')[-2], f1))

    # search_records.close()

    run_summary()