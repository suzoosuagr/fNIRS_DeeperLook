import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import random
from Tools.logger import *

"""
    New factorization of fnirs dataset
"""
class fNIRS_Basic(data.Dataset):
    def __init__(self, list_root, steps:list, mode, data_config) -> None:
        super(fNIRS_Basic, self).__init__()
        assert os.path.isdir(list_root)
        assert mode in ['train', 'eval', 'test']

        self.steps = steps
        self.mode = mode
        self.data_config = data_config
        self.data_files = []
        self.pattern_map = {
            'nb':0,
            'anb':1,
            'ewm':2,
            'rt':3,
            'gng':4,
            'es':5
        }
        self.class_map = None

    def collect_data_files(self, lines):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def get_cr_files(file):
        """
        in put the oxyfile of data
        return the control rest filename of oxy and deoxy
        """
        root = os.path.dirname(file)
        meta = os.path.basename(file).split('_')
        type, id, task, part = meta
        task = 'cr'+task
        cr_oxy = os.path.join(root, '_'.join([type, id, task, part]))
        cr_deoxy = os.path.join(root, '_'.join(['deoxy', id, task, part]))
        return cr_oxy, cr_deoxy

    @staticmethod
    def valid_label_statistic(instructor):
        static_pool = {}
        for i in instructor:
            try:
                static_pool[i[-1]]
            except KeyError:
                static_pool[i[-1]] = 0
            static_pool[i[-1]] += 1
        info(static_pool)

    @staticmethod
    def save_instructor(file_path, instructor, data_config):
        head_msg = "train-ids: {} ==== eval_ids: {} \n".format(data_config['train_ids'], data_config['eval_ids'])
        with open(file_path, 'w') as f:
            f.write(head_msg)
            for line in instructor:
                line = [str(i) for i in line]
                msg = ','.join(line) + '\n'
                f.write(msg)

    @staticmethod
    def read_instructor(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        lines = lines[1:] # skip the head message
        ins = []
        for l in lines:
            l = l.rstrip()
            ins.append(l.split(','))
        f.close()
        return ins

    @staticmethod
    def gen_2d(oxy, deoxy):
        data = np.stack([oxy, deoxy], axis=1)
        zero = np.zeros((oxy.shape[0], 2))
        data_0 = data[:,:,0:10]
        data_1 = data[:,:,10:21]
        data_2 = data[:,:,21:31]
        data_3 = data[:,:,31:42]
        data_4 = data[:,:,42:]

        data_0 = np.stack([data_0[:,:,i//2] if i%2 != 0 else zero for i in range(21)], axis=2)
        data_2 = np.stack([data_2[:,:,i//2] if i%2 != 0 else zero for i in range(21)], axis=2)
        data_4 = np.stack([data_4[:,:,i//2] if i%2 != 0 else zero for i in range(21)], axis=2)
    
        data_1 = np.stack([data_1[:,:,i//2] if i%2 == 0 else zero for i in range(21)], axis=2)
        data_3 = np.stack([data_3[:,:,i//2] if i%2 == 0 else zero for i in range(21)], axis=2)

        data = np.stack([data_0, data_1, data_2, data_3, data_4], axis=2)
        # padding:
        data = F.pad(data, (0, 1, 0, 1), 'constant', 0)
        return data 

    def get_cr_task_self_supervised(self, index):
        assert len(self.instructor) > 0
        

    def label_balance_init_instructor(self):
        """
            balance the sample step/ratio base on label
        """
        task_ratio = {
            'anb':1,
            'ewm':1,
            'rt':1,
            'gng':1
        }
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
            if stp_l in [1]:
                step = round(step/3)
            else:
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




class fNIRS_mb_label_balance_leave_subject_sla(fNIRS_Basic):
    """
        multi label
        label balance
        leave subject
        self supervised labeling 
    """
    def __init__(self, list_root, steps: list, mode, data_config, runtime=False, fold_id=0) -> None:
        super(fNIRS_mb_label_balance_leave_subject_sla, self).__init__(list_root, steps, mode, data_config)
        self.class_map = {
            "nb":[2,1],
            "anb":[2,0],
            "rt":[0,1],
            'gng':[0,1],
            'ewm':[1,2],
            'es':[0,1],
        }
        ensure(data_config['ins_root'])
        ins_path = os.path.join(data_config['ins_root'], 'M', f'{mode}_{fold_id}.txt')
        for sess in self.data_config["sessions"]:
            self.collect_data_files(os.path.join(list_root, sess+'.txt'))
        if not runtime:
            self.label_balance_init_instructor()
            self.save_instructor(ins_path, self.instructor, data_config)
        else:
            self.instructor = self.read_instructor(ins_path)

    def collect_data_files(self, file_path):
        f = open(os.path.join(file_path), 'r')
        lines = f.readlines()
        f.close()
        data_root, _ = os.path.split(lines[0].rstrip())
        for l in lines:
            l = l.rstrip()
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