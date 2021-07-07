from Tools.utils import get_files
from Tools import env_init
from Tools.logger import *
import argparse
from Experiments.Config.issue04 import *
from Data.Dataset.fnirs_finger_tapping import fNIRS_FingerTap_mb_K_fold_sla
import torch.utils.data as data 
from Model.models import BiGRU_Attn_Multi_Branch_SLA
import torch.optim as optim
from Tools.engine import fNIRS_Engine
import random
import numpy as np
import json

from Tools.metric import Performance_Test_ensemble_multi
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest="mode", default="debug", type=str)
    parser.add_argument("-l", "--log", dest="logfile", default='debug.log', type=str)
    
    return parser.parse_args()

# initialization
parser = parse_args()
args = EXP01(parser.mode, parser.logfile)
warning("STARTING >>>>>> {} ".format(args.name))
args.logpath = os.path.join(args.log_root, args.name, args.logfile)
ngpu, device, writer = env_init(args, logging.INFO)
args.ngpu = ngpu

if args.mode == 'debug':
    # args.data_config['train_ids'] = args.data_config['ids'].copy()
    # args.data_config['train_ids'].remove('2004')
    # args.data_config['eval_ids'] = ['2004']
    args.summary = False
    args.resume = False

def update_loader(fold_id, args_, test_shuffle=False):
    train_dataset = fNIRS_FingerTap_mb_K_fold_sla(\
            list_root = args_.list_path,
            steps = args_.steps_sizes,  
            mode='train',
            data_config=args_.data_config,
            runtime=True,
            fold_id=fold_id)
    eval_dataset = fNIRS_FingerTap_mb_K_fold_sla(\
            list_root = args_.list_path,
            steps = args_.steps_sizes,
            mode='eval',
            data_config=args_.data_config,
            runtime=True,
            fold_id=fold_id)
    test_dataset = fNIRS_FingerTap_mb_K_fold_sla(\
            list_root = args_.list_path,
            steps = args_.steps_sizes,
            mode='test',
            data_config=args_.data_config,
            runtime=True,
            fold_id=fold_id)

    train_loader = data.DataLoader(train_dataset, batch_size=args_.batch_size, shuffle=True, drop_last=args_.drop_last)
    eval_loader = data.DataLoader(eval_dataset, batch_size=args_.batch_size, shuffle=False, drop_last=args_.drop_last)
    test_loader = data.DataLoader(test_dataset, batch_size=args_.batch_size, shuffle=test_shuffle, drop_last=args_.drop_last)
    # debug_sample = train_dataset[0]
    return train_loader, eval_loader, test_loader

def update_model(model_name):
    if model_name == 'BiGRU_Attn_Multi_Branch_SLA':
        model = BiGRU_Attn_Multi_Branch_SLA(2, 16, 8, 6, nn.BatchNorm2d)
    else:
        raise NotImplementedError
    return model

def downsample_instructors(ins_root, fold, scheme, new_root, ratio=0.41):
    assert os.path.exists(ins_root)
    assert fold in ['22Folds', '10Folds', '5Folds']
    assert scheme in ['M', 'B']

    ins_files = get_files(os.path.join(ins_root, fold, scheme), extension_filter='.txt')
    save_folder = os.path.join(new_root, fold, scheme)
    ensure(save_folder)
    for f in ins_files:
        basename = os.path.basename(f)
        new_file_writer = open(os.path.join(save_folder, basename), 'w')
        with open(f, 'r') as fp:
            lines = fp.readlines()
            head = lines[0]
            lines = lines[1:] # skip the first line
            random.shuffle(lines)
            total_num = len(lines)
            anchor = int(total_num * ratio)
            msg_q = lines[:anchor]
            new_file_writer.write(head)
            for msg in msg_q:
                new_file_writer.write(msg)
        new_file_writer.close()
        print("converted: {}".format(basename))
    print("Done")

def generate_instructors(args):
    args_ = args
    IDS = args.data_config["ids"].copy()
    for i, id in enumerate(IDS):
        print("generating instructors ... ")
        print(f"[{i}/{len(IDS)}]managing {id}")
        args_.data_config['train_ids'] = args_.data_config['ids'].copy()
        args_.data_config['train_ids'].remove(id)
        args_.data_config['eval_ids'] = [id]

        # Dataset
        train_dataset = fNIRS_FingerTap_mb_K_fold_sla(\
                        list_root=args.list_path,
                        steps=args.steps_sizes,
                        mode='train',
                        data_config=args.data_config,
                        runtime=False,
                        fold_id=i)
        eval_dataset = fNIRS_FingerTap_mb_K_fold_sla(\
                        list_root=args.list_path,
                        steps=args.steps_sizes,
                        mode='eval',
                        data_config=args.data_config,
                        runtime=False,
                        fold_id=i)

def generate_kfold_instructors(args, kfold=5, pick=3):
    # HANDLE THE READ IDS.      
    pp_map = {}
    sample_pp_map = {}
    with open('./Data/FileList/finger_tap_ZSCORE_OXY/finger_tapping_map.txt') as ppfile:
        lines = ppfile.readlines()
        lines = [l.rstrip() for l in lines]
    for l in lines:
        p, id, _, _, _, _,_ = l.split(',')
        try:
            pp_map[id].append(p)
        except:
            pp_map[id] = [p]
    for k, v in pp_map.items():
        if len(v) > pick:
            v = v[:pick]
        else:
            pass
        sample_pp_map[k] = v

    args_ = args
    IDS = list(sample_pp_map.keys()).copy()
    random.shuffle(IDS)
    id_folds = np.array_split(IDS, kfold)
    listify = lambda folds: [list(f) for f in folds]
    fold_json = {}
    for i in range(len(id_folds)):
        id_folds_ = listify(id_folds.copy())
        print("generating instructors ... ")
        print(f"[{i}/{len(IDS)}]managing {id}")
        tem_id = id_folds_[i]
        id_folds_.remove(tem_id)
        train = []
        for ids in id_folds_:
            train += ids

        iiid_train = []
        for id in train:
            iiid_train += sample_pp_map[id]
        args_.data_config['train_ids'] = iiid_train
        iiid_eval = []
        for id in tem_id:
            iiid_eval += sample_pp_map[id]
        args_.data_config['eval_ids'] = iiid_eval
        fold_json[str(i)] = {
            'train_ids': iiid_train,
            'eval_ids': iiid_eval
        } 

        train_dataset = fNIRS_FingerTap_mb_K_fold_sla(\
                        list_root=args.list_path,
                        steps=args.steps_sizes,
                        mode='train',
                        data_config=args.data_config,
                        runtime=False,
                        fold_id=i)
        eval_dataset = fNIRS_FingerTap_mb_K_fold_sla(\
                        list_root=args.list_path,
                        steps=args.steps_sizes,
                        mode='eval',
                        data_config=args.data_config,
                        runtime=False,
                        fold_id=i)
    file_name = os.path.join(args_.data_config['ins_root'], 'fold_id_mapping.json')
    with open(file_name, 'w') as f:
        json.dump(fold_json, f)

def run_leave_subjects_out(args):
    count = 0
    accu = 0
    Basic_Name = args.name
    IDS = args.data_config["ids"].copy()
    for i, id in enumerate(IDS):
        args.data_config['train_ids'] = args.data_config['ids'].copy()
        args.data_config['train_ids'].remove(id)
        args.data_config['eval_ids'] = [id]
        args.name = "{}_{:02}".format(Basic_Name, i)
        info(f"Runing {args.name} | eval ids : {args.data_config['eval_ids']}")

        train_loader, eval_loader, test_loader = update_loader(i, args)
        model = update_model(args.model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        exe = fNIRS_Engine(train_loader, eval_loader, None, args, writer, device)
        exe.train(model, optimizer, criterion, None)

def esemble_test_subjects_out(args):
    count = 0
    accu = 0
    Basic_Name = args.name
    last = False
    ensemble_metric = Performance_Test_ensemble_multi(joint=True, self_supervise=True)
    IDS = args.data_config["ids"].copy()
    for i, id in enumerate(IDS):
        args.data_config['train_ids'] = args.data_config['ids'].copy()
        args.data_config['train_ids'].remove(id)
        args.data_config['eval_ids'] = [id]
        args.name = "{}_{:02}".format(Basic_Name, i)
        info(f"Runing {args.name} | eval ids : {args.data_config['eval_ids']}")

        train_loader, eval_loader, test_loader = update_loader(i, args)
        model = update_model(args.model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        exe = fNIRS_Engine(train_loader, eval_loader, test_loader, args, writer, device)
        # exe.train(model, optimizer, criterion, None)
        if i == len(IDS) - 1:
            last=True
        else:
            last=False
        ensemble_metric = exe.test(model, optimizer, ensemble_metric, last=last)

def esemble_test_kfolds(args, k=5):
    count = 0
    accu = 0
    Basic_Name = args.name
    # IDS = args.data_config["ids"].copy()
    ensemble_metric = Performance_Test_ensemble_multi(joint=True, self_supervise=True)
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping = json.load(jsf)
    for i in range(k):  
        # if i not in [0, k-1]:
        #     continue        # debug wise
        args.data_config['train_ids'] = fold_id_mapping[str(i)]['train_ids']
        args.data_config['eval_ids'] = fold_id_mapping[str(i)]['eval_ids']
        args.name = "{}_{:02}".format(Basic_Name, i)
        info(f"Runing {args.name} | eval on {args.data_config['eval_ids']}")

        train_loader, eval_loader, test_loader = update_loader(i, args)
        model = update_model(args.model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        exe = fNIRS_Engine(train_loader, eval_loader, test_loader, args, writer, device)
        if i == k-1:
            last=True
        else:
            last=False
        ensemble_metric = exe.test(model, optimizer, ensemble_metric, last=last)

def run_kfolds(args, k=5):
    Basic_Name = args.name
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping = json.load(jsf)
    for i in range(k):
        args.data_config['train_ids'] = fold_id_mapping[str(i)]['train_ids']
        args.data_config['eval_ids'] = fold_id_mapping[str(i)]['eval_ids']
        args.name = "{}_{:02}".format(Basic_Name, i)
        info(f"Runing {args.name} | eval on {args.data_config['eval_ids']}")

        train_loader, eval_loader, test_loader = update_loader(i, args)
        model = update_model(args.model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        exe = fNIRS_Engine(train_loader, eval_loader, None, args, writer, device)
        exe.train(model, optimizer, criterion, None)

def shap_leave_one_out(args, proc='wml'):
    info(f"PROCESSING {proc.upper()}")
    Basic_Name = args.name
    IDS = args.data_config["ids"].copy()
    for i, id in enumerate(IDS):
        args.data_config['train_ids'] = args.data_config['ids'].copy()
        args.data_config['train_ids'].remove(id)
        args.data_config['eval_ids'] = [id]
        args.name = "{}_{:02}".format(Basic_Name, i)
        info(f"Runing {args.name} | eval ids : {args.data_config['eval_ids']}")

        train_loader, eval_loader, test_loader = update_loader(i, args, test_shuffle=True)
        model = update_model(args.model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        exe = fNIRS_Engine(train_loader, eval_loader, test_loader, args, writer, device)
        exe.shap(model, optimizer, proc=proc, index=i)

if __name__ == "__main__":
    # generate_instructors(args)
    # generate_kfold_instructors(args, kfold=5, pick=args.data_config['pick']) 
    # run_leave_subjects_out(args)
    # run_kfolds(args, k=5)
    esemble_test_kfolds(args, k=5)
    # esemble_test_subjects_out(args)
    # shap_leave_one_out(args, proc='wml')
    # shap_leave_one_out(args, proc='vpl')

    # downsample_instructors('./Data/Ins/label_balance_none_zscore/', '10Folds', 'M', './Data/Ins/label_balance_sub_none_zscore/')


