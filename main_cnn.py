from torch.utils.data import dataset
from Tools.metric import PerformanceTestEnsembleMB
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.npyio import load
from Data.Dataset.fnirs import CNN_mb_label_balance_WML, CNN_mb_label_balance_VPL
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
from Model.models import CNN1
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest="test", default="debug", type=str)
    parser.add_argument("-l", "--log", dest="logfile", default='debug.log', type=str)
    
    return parser.parse_args()

parser = parse_args()
args = EXP03('debug', 'standard.log')
warning("STARTING >>>>>> {} ".format(args.name))
args.logpath = os.path.join(args.log_root, args.name, args.logfile)
ngpu, device, writer = env_init(args, logging.INFO)
args.ngpu = ngpu

def update_loader(fold_id, args_):
    train_dataset = CNN_mb_label_balance_VPL(
            list_root = args_.list_path,
            steps = args_.steps_sizes,  
            mode='train',
            data_config=args_.data_config,
            runtime=True,
            fold_id=fold_id)
    eval_dataset  = CNN_mb_label_balance_VPL(
            list_root = args_.list_path,
            steps = args_.steps_sizes,  
            mode='eval',
            data_config=args_.data_config,
            runtime=True,
            fold_id=fold_id)
    test_dataset  = CNN_mb_label_balance_VPL(
            list_root = args_.list_path,
            steps = args_.steps_sizes,  
            mode='test',
            data_config=args_.data_config,
            runtime=True,
            fold_id=fold_id)

    # debug
    train_sample = train_dataset[0]

    train_loader = data.DataLoader(train_dataset, batch_size=args_.batch_size, shuffle=True, drop_last=args_.drop_last)
    eval_loader = data.DataLoader(eval_dataset, batch_size=args_.batch_size, shuffle=False, drop_last=args_.drop_last)
    test_loader = data.DataLoader(test_dataset, batch_size=args_.batch_size, shuffle=False, drop_last=args_.drop_last)
    return train_loader, eval_loader, test_loader

def update_model(model_name, filterList=[32]):
    if model_name == 'CNN1':
        model = CNN1(104, filterList, 3)
    # elif model_name == 'CNN2':
    #     # model = CNN
    else:
        raise NotImplementedError
    return model

def run_kfolds(args, filterList, k=5, mode='test'):
    Basic_Name = args.name
    ensemble_metrics = PerformanceTestEnsembleMB(size=3)
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping = json.load(jsf)
    for fold_id in range(k):
        args.data_config['train_ids'] = fold_id_mapping[str(fold_id)]['train_ids']
        args.data_config['eval_ids'] = fold_id_mapping[str(fold_id)]['eval_ids']
        args.name = "{}_{:02}".format(Basic_Name, fold_id)
        info(f"Runing {args.name} | eval on {args.data_config['eval_ids']}")

        train_loader, eval_loader, test_loader = update_loader(fold_id, args)
        model = update_model(args.model, filterList).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        exe = Ann_Engine(train_loader, eval_loader, test_loader, args, writer, device)
        if mode in ['train', 'full']:
            exe.train(model, optimizer, criterion, None)
        if mode in ['test', 'full']:
            if fold_id == k-1:
                last=True
            else:
                last=False
            ensemble_metrics = exe.ensemble_test(model, optimizer, ensemble_metrics, Basic_Name, last=last)
    return ensemble_metrics.value()

if __name__ == '__main__':
    basic_name = args.name
    search_records = open('./Files/cnn/CNN1_10FoldsCV_Metric_VPL.txt', 'w')
    for fL in [[32]]:
        confMat, accu, precision, recall, f1 = run_kfolds(args, fL, k=10, mode='full')
        search_records.write("+"*20+'\n')
        search_records.write("ACCURACY_{} = {}\n".format(fL, accu))
        search_records.write("Precision{} = {}\n".format(fL, precision))
        search_records.write("Recall_{} = {}\n".format(fL, recall))
        search_records.write("F1_{} = {}\n".format(fL, f1))