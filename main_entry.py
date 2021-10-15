import torchsummary
from Tools.utils import get_files
from Tools import env_init
from Tools.logger import *
import argparse
from Experiments.Config.issue01 import *
from Data.Dataset.fnirs import fNIRS_mb_label_balance_leave_subject_sla
import torch.utils.data as data 
from Model.models import BiGRU_Attn_Multi_Branch_SLA
import torch.optim as optim
from Tools.engine import fNIRS_Engine
import random
import numpy as np
import json
import scipy.stats as st
from Tools.metric import Performance_Test_ensemble_multi
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datetime
from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest="mode", default="debug", type=str)
    parser.add_argument("-l", "--log", dest="logfile", default='debug.log', type=str)
    parser.add_argument('--exp', default='EXP04', type=str)
    return parser.parse_args()

# initialization
parser = parse_args()
args = eval(parser.exp)(parser.mode, parser.logfile)
warning("STARTING >>>>>> {} ".format(args.name))
args.logpath = os.path.join(args.log_root, args.name, args.logfile)
ngpu, device, _ = env_init(args, logging.INFO)
args.ngpu = ngpu

if args.mode == 'debug':
    # args.data_config['train_ids'] = args.data_config['ids'].copy()
    # args.data_config['train_ids'].remove('2004')
    # args.data_config['eval_ids'] = ['2004']
    args.summary = False
    args.resume = False

def update_loader(fold_id, args_, test_shuffle=False):
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
    test_loader = data.DataLoader(test_dataset, batch_size=args_.batch_size, shuffle=test_shuffle, drop_last=args_.drop_last)

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
        ensemble_metric = exe.test(model, optimizer, ensemble_metric, mask=None, last=last)

def visual_importance_kfolds(args, k=5):
    baseline = {
        'wml': [0.7753, 0.7741, 0.7765, 0.7742],
        'vpl': [0.7730, 0.7722, 0.7742, 0.7741]
    }

    Basic_Name = args.name
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping = json.load(jsf)
    maskList = []
    for i in range(2):
        r = i*2
        for j in range(10):
            l = j*2+1
            mask = [(r, l), (r+1, l-1), (r+1, l+1), (r+2, l)]
            maskList.append(mask)

    # for j in range(10):
    #     l = j*2+1
    #     r=0
    #     mask = [(r, l), (r+1, l-1), (r+1, l+1), (r+2, l)]
    #     r=2
    #     mask += [(r+1, l-1), (r+1, l+1), (r+2, l)]
    #     maskList.append(mask)
    
    # maskList = [[(2, 11), (3, 10), (3, 12), (4, 11), (2, 15), (3, 14), (3, 16), (4, 15), (2, 13), (3, 12), (3, 14), (4, 13), (0, 7), (1, 6), (1, 8), (2, 7), (0, 9), (1, 8), (1, 10), (2, 9)]]
    
    rwriter = open(f'./Files/ImpactAnalysis/{Basic_Name}_Impact_Diamond_4ch_replace0.txt', 'w')

    for mask in maskList:
        ensemble_metric = Performance_Test_ensemble_multi(joint=True, self_supervise=True)
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
            ensemble_metric = exe.test(model, optimizer, ensemble_metric, mask=mask, last=last)
        performance = ensemble_metric.value()
        rwriter.write("+"*20+f"mask: {mask}"+'\n')
        for ttt in ['wml', 'vpl']:
            a, p, r, f = baseline[ttt]
            conf_mat, accu, precision, recall, f1 = performance[ttt]
            rwriter.write("ACCURACY_{} = {}\n".format(ttt, accu))
            rwriter.write("Precision{} = {}\n".format(ttt, precision))
            rwriter.write("Recall_{} = {}\n".format(ttt, recall))
            rwriter.write("F1_{} = {}\n".format(ttt, f1))

            rwriter.write(f"Drop_{ttt.upper()}: accu        {np.mean(accu) - a}\n")
            rwriter.write(f"Drop_{ttt.upper()}: precision   {np.mean(precision) - p}\n")
            rwriter.write(f"Drop_{ttt.upper()}: recall      {np.mean(recall) - r}\n")
            rwriter.write(f"Drop_{ttt.upper()}: f1      {np.mean(f1) - f}\n")


    rwriter.close()




def fold_test(args, k=10):
    count = 0
    Basic_Name = args.name
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping = json.load(jsf)
    for i in range(k):
        metric = Performance_Test_ensemble_multi(joint=True, self_supervise=True)
        args.data_config['train_ids'] = fold_id_mapping[str(i)]['train_ids']
        args.data_config['eval_ids'] = fold_id_mapping[str(i)]['eval_ids']
        args.name = "{}_{:02}".format(Basic_Name, i)
        info(f"Runing {args.name} | eval on {args.data_config['eval_ids']}")

        train_loader, eval_loader, test_loader = update_loader(i, args)
        model = update_model(args.model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        exe = fNIRS_Engine(train_loader, eval_loader, test_loader, args, writer, device)
        info(f'fold id is {i}')
        exe.test(model, optimizer, metric, last=True)

def run_kfolds(args, k=5):
    count = 0
    accu = 0
    Basic_Name = args.name
    IDS = args.data_config["ids"].copy()
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping = json.load(jsf)
    for i in range(k):
        if args.summary:
            if args.mode in args.summary_register:
                ensure(args.summary_dir)
                summary_dir = os.path.join(args.summary_dir, args.name+f'_{i}'+'/',datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
                writer = SummaryWriter(summary_dir)
                info("tfboard writer created. ")
            else:
                writer = None
        else:
            writer = None

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

def run_pytorchsummary(args):
    model = update_model(args.model).to(device)
    psummary = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of parameters {}".format(psummary))
    

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

def shap_kfolds(args, k, proc='wml'):
    info(f"PROCESSING {proc.upper()}")
    Basic_Name = args.name
    with open(os.path.join(args.data_config['ins_root'], 'fold_id_mapping.json'), 'r') as jsf:
        fold_id_mapping  = json.load(jsf)
    for fold_id in range(k):
        args.data_config['train_ids'] = fold_id_mapping[str(fold_id)]['train_ids']
        args.data_config['eval_ids'] = fold_id_mapping[str(fold_id)]['eval_ids']
        args.name = "{}_{:02}".format(Basic_Name, fold_id)
        info(f"Runing {args.name} | eval on {args.data_config['eval_ids']}")

        train_loader, eval_loader, test_loader = update_loader(fold_id, args, test_shuffle=True)
        model = update_model(args.model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        exe = fNIRS_Engine(train_loader, eval_loader, test_loader, args, writer, device)
        exe.shap(model, optimizer, proc=proc, index=fold_id)


if __name__ == "__main__":
    # generate_instructors(args)
    # generate_kfold_instructors(args, k=10)
    # run_leave_subjects_out(args)
    # run_kfolds(args, k=10)
    run_pytorchsummary(args)
    # esemble_test_kfolds(args, k=10)
    # visual_importance_kfolds(args, k=10)
    # fold_test(args, k=10)
    # esemble_test_subjects_out(args)
    # shap_leave_one_out(args, proc='wml')
    # shap_leave_one_out(args, proc='vpl')

    # downsample_instructors('./Data/Ins/label_balance_none_zscore/', '10Folds', 'M', './Data/Ins/label_balance_sub_none_zscore/')
    # shap_kfolds(args, 10, 'wml')
    # shap_kfolds(args, 10, 'vpl')

    



