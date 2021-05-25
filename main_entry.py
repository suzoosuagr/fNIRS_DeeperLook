from Tools import env_init
from Tools.logger import *
import argparse
from Experiments.Config.issue01 import *
from Data.Dataset.fnirs import fNIRS_mb_label_balance_leave_subject_sla
from Experiments.Config.issue01 import *
import torch.utils.data as data 
from Model.models import BiGRU_Attn_Multi_Branch_SLA
import torch.optim as optim
from Tools.engine import fNIRS_Engine
import random
import numpy as np
import json

from Tools.metric import Performance_Test_ensemble_multi

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest="mode", default="debug", type=str)
    parser.add_argument("-l", "--log", dest="logfile", default='debug.log', type=str)
    
    return parser.parse_args()

# initialization
parser = parse_args()
args = EXP02(parser.mode, parser.logfile)
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

def update_loader(fold_id, args_):
    train_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
            list_root = args_.list_path,
            steps = args_.steps_sizes,
            mode='train',
            data_config=args_.data_config,
            runtime=True,
            fold_id=fold_id)
    eval_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
            list_root = args_.list_path,
            steps = args_.steps_sizes,
            mode='eval',
            data_config=args_.data_config,
            runtime=True,
            fold_id=fold_id)
    test_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
            list_root = args_.list_path,
            steps = args_.steps_sizes,
            mode='test',
            data_config=args_.data_config,
            runtime=True,
            fold_id=fold_id)

    train_loader = data.DataLoader(train_dataset, batch_size=args_.batch_size, shuffle=True, drop_last=args_.drop_last)
    eval_loader = data.DataLoader(eval_dataset, batch_size=args_.batch_size, shuffle=False, drop_last=args_.drop_last)
    test_loader = data.DataLoader(test_dataset, batch_size=args_.batch_size, shuffle=False, drop_last=args_.drop_last)

    return train_loader, eval_loader, test_loader

def update_model(model_name):
    if model_name == 'BiGRU_Attn_Multi_Branch_SLA':
        model = BiGRU_Attn_Multi_Branch_SLA(2, 16, 8, 6, nn.BatchNorm2d)
    else:
        raise NotImplementedError
    return model

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
        train_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
                        list_root=args.list_path,
                        steps=args.steps_sizes,
                        mode='train',
                        data_config=args.data_config,
                        runtime=False,
                        fold_id=i)
        eval_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
                        list_root=args.list_path,
                        steps=args.steps_sizes,
                        mode='eval',
                        data_config=args.data_config,
                        runtime=False,
                        fold_id=i)

def generate_kfold_instructors(args, k=5):
    args_ = args
    IDS = args.data_config["ids"].copy()
    random.shuffle(IDS)
    id_folds = np.array_split(IDS, k)
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

        args_.data_config['train_ids'] = train
        args_.data_config['eval_ids'] = tem_id
        fold_json[str(i)] = {
            'train_ids': train,
            'eval_ids': tem_id
        } 

        train_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
                        list_root=args.list_path,
                        steps=args.steps_sizes,
                        mode='train',
                        data_config=args.data_config,
                        runtime=False,
                        fold_id=i)
        eval_dataset = fNIRS_mb_label_balance_leave_subject_sla(\
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
        exe.test(model, optimizer, ensemble_metric, last=last)

def run_kfolds(args, k=5):
    count = 0
    accu = 0
    Basic_Name = args.name
    IDS = args.data_config["ids"].copy()
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

if __name__ == "__main__":
    # generate_instructors(args)
    # generate_kfold_instructors(args, k=5)
    # run_leave_subjects_out(args)
    run_kfolds(args, k=5)
    # esemble_test_subjects_out(args)

