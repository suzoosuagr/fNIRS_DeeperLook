import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import random

from torch.utils.data.dataset import Dataset
from Tools.logger import *
from Data.Dataset.fnirs import fNIRS_Basic

class fNIRS_FingerTap_mb_K_fold_sla(fNIRS_Basic):
    """
        multi label, 
        label balance,
        leave subject
        sla.
    """
    def __init__(self, list_root, steps: list, mode, data_config, runtime=False, fold_id=0, istest=False) -> None:
        super(fNIRS_FingerTap_mb_K_fold_sla, self).__init__(list_root, steps, mode, data_config)

        self.pattern_map = {
            'RH':0,
            'LH':1,
            'BH':2,
            'RL':3,
            'LL':4,
            'BL':5,
        }

        self.class_map = {
            'RH':[0, 2],
            'LH':[2, 0],
            'BH':[2, 2],
            'RL':[0, 1],
            'LL':[1, 0],
            'BL':[1, 1]
        }
        self.istest = istest
        ensure(data_config['ins_root'])
        ins_path = os.path.join(data_config['ins_root'], 'M', f'{mode}_{fold_id}.txt')
        if mode in ['test']:
            ins_path = os.path.join(data_config['ins_root'], 'M', f'eval_{fold_id}.txt')
        #  only one session included
        for sess in self.data_config["sessions"]:
            self.collect_data_files(os.path.join(list_root, sess+'.txt'))
        if not runtime:
            self.label_balance_init_instructor()
            self.save_instructor(ins_path, self.instructor, data_config)
        else:
            self.instructor = self.read_instructor(ins_path)
            self.valid_label_statistic(self.instructor)
    
    def __getitem__(self, index):
        return self.get_cr_task_self_supervised_multi_branch(index)

    def __len__(self):
        return len(self.instructor)

    def gen_2d(self, oxy, deoxy):
        """
            this gen_2d only work for the finger tapping. 
        """
        data = np.stack([oxy, deoxy], axis=1)
        zero = np.zeros((oxy.shape[0], 2))
        # based on the geometry of the finger tapping data. 
        data_0 = data[:,:,[14, 12, 1, 43, 31, 29]]
        data_2 = data[:,:,[19, 8, 5, 39, 36, 26]]
        data_4 = data[:,:,[18, 15, 4, 45, 35, 32]]
        data_6 = data[:,:,[22, 11, 9, 42, 40, 28]]
        # ==== 
        data_1 = data[:,:,[20, 13, 6, 0, 44, 37, 30, 24]]
        data_3 = data[:,:,[21, 16, 7, 2, 46, 38, 33, 25]]
        data_5 = data[:,:,[23, 17, 10, 3, 47, 41, 34, 27]]

        data_0 = self.geoAssemb(data_0, zero, odd=False)
        data_2 = self.geoAssemb(data_2, zero, odd=False)
        data_4 = self.geoAssemb(data_4, zero, odd=False)
        data_6 = self.geoAssemb(data_6, zero, odd=False)
    
        data_1 = self.geoAssemb(data_1, zero, odd=True)
        data_3 = self.geoAssemb(data_3, zero, odd=True)
        data_5 = self.geoAssemb(data_5, zero, odd=True)
        

        data = np.stack([data_0, data_1, data_2, data_3, data_4, data_5, data_6], axis=2)
        data = torch.from_numpy(data).float()
        return data

    def geoAssemb(self, data, zero, odd=True):
        L, R = np.split(data, 2, axis=2)
        if odd == True:
            data_l = np.stack([L[:,:, i//2] if i%2 == 0 else zero for i in range(7)], axis=2)
            data_r = np.stack([R[:,:, i//2] if i%2 == 0 else zero for i in range(7)], axis=2)
        else:
            data_l = np.stack([L[:,:, i//2] if i%2 != 0 else zero for i in range(7)], axis=2)
            data_r = np.stack([R[:,:, i//2] if i%2 != 0 else zero for i in range(7)], axis=2)
        return np.concatenate([data_l, data_r], axis=2)

    def collect_data_files(self, file_path):
        with open(os.path.join(file_path), 'r') as fp:
            lines = fp.readlines()
            lines = [l.rstrip() for l in lines]

        data_root, _ = os.path.split(lines[0])
        for l in lines:
            base = os.path.basename(l)
            meta = base.split('_')
            type, id, task, part = meta
            deoxy_file = '_'.join(['deoxy', id, task, part])

            if self.mode in ["train"] and id in self.data_config['train_ids'] and task in self.data_config["train_tasks"] and part in self.data_config["parts"]:
                self.data_files.append([l, os.path.join(data_root, deoxy_file)])
            elif self.mode in ["eval", "test"] and id in self.data_config['eval_ids'] and task in self.data_config["eval_tasks"] and part in self.data_config["parts"]:
                self.data_files.append([l,os.path.join(data_root, deoxy_file)])
            else:
                continue

    def get_cr_files(self,file):
        root = os.path.dirname(file)
        meta = os.path.basename(file).split('_')
        type, id, task, part = meta
        task = 'CR'+task
        cr_oxy = os.path.join(root, '_'.join([type, id, task, part]))
        cr_deoxy = os.path.join(root, '_'.join(['deoxy', id, task, part]))
        return cr_oxy, cr_deoxy

    def  label_balance_init_instructor(self):
        """
            balance the sample step/ratio base on label
        """
        # task_ratio = {
        #     'anb':1,
        #     'ewm':1,
        #     'rt':1,
        #     'gng':1
        # }
        essemble_cr_length = [0] * 6
        essemble_task_length = [0] * 6

        for f_idx  in range(len(self.data_files)):
            # read filename
            oxy_file, deoxy_file = self.data_files[f_idx]
            cr_oxy_file, cr_deoxy_file = self.get_cr_files(oxy_file)
            _, _, task, _ = os.path.basename(oxy_file).split('_')
            plabel = self.pattern_map[task]   # plabel: rea the pattern label
            # load data
            oxy_task = np.load(oxy_file)
            oxy_cr = np.load(cr_oxy_file)
            # statistics
            essemble_cr_length[plabel] += len(oxy_cr)
            essemble_task_length[plabel] += len(oxy_task)

        max_step = self.steps[2]
        # remove zeros
        essemble_task_length_ = essemble_task_length.copy()
        # try:
        #     while True:
        #         essemble_task_length_.remove(0)
        # except ValueError:
        #     pass

        max_length = np.max(essemble_task_length_)
        tasks_steps = (essemble_task_length_ / max_length) * max_step
        tasks_steps = [int(i) for i in tasks_steps]

        instructor = []
        count = 0
        for f_idx in range(len(self.data_files)):
            # read filename
            oxy_file, deoxy_file = self.data_files[f_idx]
            cr_oxy_file, cr_deoxy_file = self.get_cr_files(oxy_file)
            _, _, task, _ = os.path.basename(oxy_file).split('_')
            stp_l = self.pattern_map[task]

            # load data
            oxy_task = np.load(oxy_file)
            oxy_cr = np.load(cr_oxy_file)
            # generate instructor:
            cr_duration, task_duration, _ = self.steps
            step = tasks_steps[stp_l]
            step = round(step)
            
            cr_len = len(oxy_cr)
            task_len = len(oxy_task)

            cr_begin = 0
            task_begin = 0
            cr_board = cr_len - cr_duration
            task_board = task_len - task_duration

            # manage position
            legi_cr = [i for i in range(cr_begin, cr_board, 25)]
            for i in range(task_begin, task_board, step):
                cr_idx = (i // step) % len(legi_cr)
                instructor.append([self.data_files[f_idx], legi_cr[cr_idx], i, task])
                count += 1
        # get the instructor
        self.instructor = instructor
        # logging the data information. 
        self.valid_label_statistic(self.instructor)

    def get_cr_task_self_supervised_multi_branch(self, index):
        cr, task, label, oxy_file = self.load_data_(index)

        # random self supervision 
        if self.mode in ["train", "eval"]:
            tr = np.random.uniform(0.0, 1.0)
            if tr > 0.5:
                label = [l*2 + 0 for l in label]
                return np.concatenate((cr, task), axis=0).astype(np.float32), label, 0, oxy_file
            else:
                label = [l*2 + 1 for l in label]
                return np.concatenate((task, cr), axis=0).astype(np.float32), label, 1, oxy_file
        elif self.mode in ["test"]:
            return [np.concatenate((cr, task), axis=0).astype(np.float32), self.MBlabeling(label, 0.5+1)], [np.concatenate((task, cr), axis=0).astype(np.float32), self.MBlabeling(label, 0.5-1)]

class CNN_mb_label_balance_WML(fNIRS_FingerTap_mb_K_fold_sla):
    def __init__(self, list_root, steps: list, mode, data_config, runtime, fold_id, istest=False) -> None:
        super(CNN_mb_label_balance_WML, self).__init__(list_root, steps, mode, data_config, runtime=runtime, fold_id=fold_id, istest=istest)
        self.class_map = {
            'RH':0,
            'LH':2,
            'BH':2,
            'RL':0,
            'LL':1,
            'BL':1
        }

    def get_cr_task_self_supervised_multi_branch(self, index):
        cr, task, label, oxy_file = self.load_data_(index, trans=False)
        return task.astype(np.float32), label, oxy_file

class fNIRS_FingerTap_mb_K_fold(fNIRS_FingerTap_mb_K_fold_sla):
    def __init__(self, list_root, steps: list, mode, data_config, runtime, fold_id, istest=False) -> None:
        super(fNIRS_FingerTap_mb_K_fold, self).__init__(list_root, steps, mode, data_config, runtime=runtime, fold_id=fold_id, istest=istest)

    def __getitem__(self, index):
        return self.get_cr_task_self_supervised_multi_branch(index)
    
    def get_cr_task_self_supervised_multi_branch(self, index):
        cr, task, label, oxy_file = self.load_data_(index)
        return task.astype(np.float32), label, oxy_file