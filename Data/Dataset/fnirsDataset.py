 import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import random


"""
fNIRS Block-wise split method. 
Binary
Only focus on one of WML or VPL
"""

class fNIRS_Basic(data.Dataset):
    """
    fNIRS dataset:
    - dynamic
    - block wise cross validation 
    - down sample. 
    - view cr as task. 
    """
    def __init__(self, list_path, steps: list, mode, data_config, out_mode=0, sampling='sliding', train_ratio=0.7, mb=False) -> None:
        """
        `steps`: list [cr_duration, task_duration, step]\\
        `data_config`: dict, describe the ids and tasks\\
        `out_mode`: int {0 for vector, 1 for matrix}\\
        `method`: str splitting by sessions or subject\\
        `sampling`: choice from [random and sliding]
        `ins_files`: dict of instructor files. 
        """
        assert os.path.isdir(list_path)
        assert mode in ['train', 'eval', 'test']
        
        self.mb = mb
        self.mode = mode
        self.out_mode = out_mode
        self.data_config = data_config
        self.steps = steps   # list [duration_cr, duration_task, sliding_step]
        self.sampling = sampling
        self.train_ratio = train_ratio
        self.instructor = None
        self.class_map = None
        self.pattern_map = {
            "nb":0,
            "anb":1,
            "ewm":2,
            "rt":3,
            "gng":4,
            "es":5
        }

        # Read Lines
        f1_list = open(os.path.join(list_path, 's1.txt'), 'r')
        f2_list = open(os.path.join(list_path, 's2.txt'), 'r')
        s1_lines = f1_list.readlines()
        s2_lines = f2_list.readlines()
        f1_list.close()
        f2_list.close()

        self.data_files = []
        # self.eval_files = []

        if self.data_config['split'] == "sessions":
            if self.mode == "train":
                eval('self.collect_data_files_{}'.format(self.data_config['split']))(s1_lines)
            elif self.mode in ["eval", "test"]:
                eval('self.collect_data_files_{}'.format(self.data_config['split']))(s2_lines)
        else:
            if "s1" in self.data_config["sessions"]:
                # self.collect_data_files_trails(s1_lines)
                eval('self.collect_data_files_{}'.format(self.data_config['split']))(s1_lines)
            if "s2" in self.data_config["sessions"]:
                eval('self.collect_data_files_{}'.format(self.data_config['split']))(s2_lines)

        
        if self.data_config["bound"] == "upper":
            self.upper_init_instructor()
        elif self.data_config["bound"] == "label":
            self.label_balance_init_instructor()
        else:
            pass
        # reading the statistical info about the data
        self.valid_label_statistic()

    def valid_label_statistic(self):
        static_pool = {}

        for i in self.instructor:
            try:
                static_pool[i[-1]]
            except KeyError:
                static_pool[i[-1]] = 0
            static_pool[i[-1]] += 1

        print("Statical Pool : {\n}")
        print(static_pool)
        
    def label_balance_init_instructor(self):
        """
        balance the sample step/ratio base on label
        """
        task_ratio = {
            'anb': 1, 
            'ewm': 1,
            'rt': 1,
            'gng': 1
        }

        essemble_cr_length = [0,0,0,0,0,0]
        essemble_task_length = [0,0,0,0,0,0]

        for f_idx in range(len(self.data_files)):
            # read filename
            oxy_file, deoxy_file = self.data_files[f_idx]
            cr_oxy_file, cr_deoxy_file = self.get_cr_files(oxy_file)
            _, _, task, _= os.path.basename(oxy_file).split('_')
            label = self.pattern_map[task]

            # load data
            oxy_task = np.load(oxy_file)
            oxy_cr = np.load(cr_oxy_file)

            # statistics
            essemble_cr_length[label] += len(oxy_cr)
            essemble_task_length[label] += len(oxy_task)

        max_step = self.steps[2]
        
        # remove zeros
        essemble_task_length_ = essemble_task_length.copy()
        try:
            while True:
                essemble_task_length_.remove(0)
        except ValueError:
            pass

        # min_length = np.min(esemble_task_length[1:])
        max_length = np.max(essemble_task_length_)
        tasks_steps = (essemble_task_length / max_length) * max_step
        tasks_steps = [i for i in tasks_steps]
        
        # instructor should be a list for shuffle purpose. 
        instructor = []
        count = 0
        for f_idx in range(len(self.data_files)):
            # read filename
            oxy_file, deoxy_file = self.data_files[f_idx]
            cr_oxy_file, cr_deoxy_file = self.get_cr_files(oxy_file)
            _, _, task, _ = os.path.basename(oxy_file).split('_')
            label = self.class_map[task]
            stp_l = self.pattern_map[task]

            # load data
            oxy_task = np.load(oxy_file)
            oxy_cr = np.load(cr_oxy_file)

            # generate instructor:  
            cr_duration, task_duration, _ = self.steps
            step = tasks_steps[stp_l]
            if stp_l in [1, 2]:
                step = int(step / 2)
            else:
                step = int(step)
            
            cr_len = len(oxy_cr)
            task_len = len(oxy_task)
            cr_board = cr_len - cr_duration
            task_board = task_len - task_duration
            
            # Mode:
            if self.mode in ['train']:
                cr_begin = 0
                task_begin = 0
                cr_board = int(cr_board * self.train_ratio) - 50
                task_board = int(task_board * self.train_ratio) - 50
            elif self.mode in ['eval', 'test']:
                cr_begin = int(cr_board * self.train_ratio)
                task_begin = int(task_board * self.train_ratio)

            # manage position
            legi_cr = [i for i in range(cr_begin, cr_board, 25)]  # fixed cr step
            # legi_cr = [i for i in range(cr_begin, cr_board, step)] 
            for i in range(task_begin, task_board, step):
                cr_idx = (i // step) % len(legi_cr) 
                instructor.append([f_idx, legi_cr[cr_idx], i, task])
                count += 1

        self.instructor = instructor
        self.rescale_instructor(ratio=4/6)

    def rescale_instructor(self, ratio=4/6):
        static_pool = {}
        rescale_ins = []
        for i in self.instructor:
            prob = np.random.uniform(0.0, 1.0)
            if prob < ratio:
                rescale_ins.append(i)
        self.instructor = rescale_ins
        for i in self.instructor:
            try:
                static_pool[i[-1]]
            except KeyError:
                static_pool[i[-1]] = 0
            static_pool[i[-1]] += 1
        num_anb = static_pool['anb']
        ratio_ = 4000  / num_anb
        rescale_ins=[]
        for r in self.instructor:
            if r[-1] in ['anb', 'ewm']:
                prob = np.random.uniform(0.0, 1.0)
                if prob < ratio_:
                    rescale_ins.append(r)
            else:
                rescale_ins.append(r)
        self.instructor = rescale_ins

    def upper_init_instructor(self):
        """
        This is for dynamically generate the instructor
        using different sliding step for different tasks and let final data balanced. 
        Decrease the step size for shorter task. 
        """

        essemble_cr_length = [0,0,0,0,0,0]
        essemble_task_length = [0,0,0,0,0,0]

        for f_idx in range(len(self.data_files)):
            # read filename
            oxy_file, deoxy_file = self.data_files[f_idx]
            cr_oxy_file, cr_deoxy_file = self.get_cr_files(oxy_file)
            _, _, task, _= os.path.basename(oxy_file).split('_')
            label = self.pattern_map[task]

            # load data
            oxy_task = np.load(oxy_file)
            oxy_cr = np.load(cr_oxy_file)

            # statistics
            essemble_cr_length[label] += len(oxy_cr)
            essemble_task_length[label] += len(oxy_task)

        max_step = self.steps[2]
        
        # remove zeros
        essemble_task_length_ = essemble_task_length.copy()
        try:
            while True:
                essemble_task_length_.remove(0)
        except ValueError:
            pass

        # min_length = np.min(esemble_task_length[1:])
        max_length = np.max(essemble_task_length_)
        tasks_steps = (essemble_task_length / max_length) * max_step
        tasks_steps = [int(i) for i in tasks_steps]
        
        # instructor should be a list for shuffle purpose. 
        instructor = []
        count = 0
        for f_idx in range(len(self.data_files)):
            # read filename
            oxy_file, deoxy_file = self.data_files[f_idx]
            cr_oxy_file, cr_deoxy_file = self.get_cr_files(oxy_file)
            _, _, task, _ = os.path.basename(oxy_file).split('_')
            label = self.class_map[task]
            stp_l = self.pattern_map[task]

            # load data
            oxy_task = np.load(oxy_file)
            oxy_cr = np.load(cr_oxy_file)

            # generate instructor:  
            cr_duration, task_duration, _ = self.steps
            step = tasks_steps[stp_l]
            
            cr_len = len(oxy_cr)
            task_len = len(oxy_task)
            cr_board = cr_len - cr_duration
            task_board = task_len - task_duration
            
            # Mode:
            if self.mode in ['train']:
                cr_begin = 0
                task_begin = 0
                cr_board = int(cr_board * self.train_ratio) - 50
                task_board = int(task_board * self.train_ratio) - 50
            elif self.mode in ['eval', 'test']:
                cr_begin = int(cr_board * self.train_ratio)
                task_begin = int(task_board * self.train_ratio)

            # manage position
            legi_cr = [i for i in range(cr_begin, cr_board, 25)]  # fixed cr step
            # legi_cr = [i for i in range(cr_begin, cr_board, step)] 
            for i in range(task_begin, task_board, step):
                cr_idx = (i // step) % len(legi_cr) 
                instructor.append([f_idx, legi_cr[cr_idx], i, task])
                count += 1

        self.instructor = instructor


        print("Reverse DYNAMIC FINISHED PREPARE IDX INSTRUCTOR OF FNIRS FOR {}".format(self.mode.upper()))

    def save_instructor(self, filename, ins):
        with open(filename, 'w') as f:
            for line in ins:
                line = [str(i) for i in line]
                str_line = ','.join(line)+'\n'
                f.write(str_line)
            
    def read_instructor(self, filename):
        """
        read from file to self.instructor
        """
        f = open(filename, 'r') 
        lines = f.readlines()
        ins = []
        for l in lines:
            l = l.rstrip()
            ins.append(l.split(','))
        f.close()

        self.instructor = ins
    
    def __getitem__(self, index):
        if self.sampling == 'cr_task':
            return self.get_cr_task_self_supervised(index)
        elif self.sampling == 'task':
            return self.get_task(index)
        else:
            raise NotImplementedError 

    def get_cr_task_self_supervised(self, index):
        assert len(self.instructor) > 0
        f_idx, start_cr, start_task, label = self.instructor[index]
        start_cr = int(start_cr)
        start_task = int(start_task)

        label = self.class_map[label]
        _, task_duration, _ = self.steps

        # read filename
        oxy_file, deoxy_file = self.data_files[f_idx]
        cr_oxy_file, cr_deoxy_file = self.get_cr_files(oxy_file)

        #load_data
        oxy_task = np.load(oxy_file)
        oxy_cr = np.load(cr_oxy_file)

        deoxy_task = np.load(deoxy_file)
        deoxy_cr = np.load(cr_deoxy_file)

        # slice
        oxy_cr_sample = oxy_cr[start_cr: start_cr + task_duration]
        oxy_task_sample = oxy_task[start_task: start_task + task_duration]

        deoxy_cr_sample = deoxy_cr[start_cr: start_cr + task_duration]
        deoxy_task_sample = deoxy_task[start_cr: start_cr + task_duration]

        cr = self.gen_2d(oxy_cr_sample, deoxy_cr_sample)
        task = self.gen_2d(oxy_task_sample, deoxy_task_sample)

        # random 
        if not self.mb:
            if self.mode in ["train", "eval"]:
                tr = np.random.uniform(0.0, 1.0)
                if tr > 0.5:
                    label = label * 2 + 0
                    
                    return np.concatenate((cr, task), axis=0).astype(np.float32), label, 0, oxy_file 
                else:
                    label = label * 2 + 1
                    return np.concatenate((task, cr), axis=0).astype(np.float32), label, 1, oxy_file

            elif self.mode in ["test"]:
                return [np.concatenate((cr, task), axis=0).astype(np.float32), label*2+0], [np.concatenate((task, cr), axis=0).astype(np.float32), label*2+1]
        else:
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

    def MBlabeling(self, label, tr):
        if tr>0.5:
            return [l*2 + 0 for l in label]
        else:
            return [l*2 + 1 for l in label]

    def get_task(self, index):
        assert len(self.instructor) > 0
        f_idx, _, start_task, label = self.instructor[index]
        f_idx = int(f_idx)
        # start_cr = int(start_cr)
        start_task = int(start_task)
        label = self.class_map[label]
        _, task_duration, _ = self.steps
        # read filename
        oxy_file, deoxy_file = self.data_files[f_idx]
        # cr_oxy_file, cr_deoxy_file = self.get_cr_files(oxy_file)

        # load data
        oxy_task = np.load(oxy_file)
        # oxy_cr = np.load(cr_oxy_file)

        deoxy_task = np.load(deoxy_file)
        # deoxy_cr = np.load(cr_deoxy_file)

        # slice:
        # oxy_cr_sample = oxy_cr[start_cr: start_cr + cr_duration]
        oxy_task_sample = oxy_task[start_task: start_task + task_duration]
        # deoxy_cr_sample = deoxy_cr[start_cr: start_cr + cr_duration]
        deoxy_task_sample = deoxy_task[start_task: start_task + task_duration]

        if self.out_mode == 0:
            # cr = np.concatenate((oxy_cr_sample, deoxy_cr_sample), axis=0)      #  [cr_len*2, num_channels]
            task = np.concatenate((oxy_task_sample, deoxy_task_sample), axis=0)   #  [task_len*2, num_channels]
        elif self.out_mode == 1:
            # cr = self.gen_2d(oxy_cr_sample, deoxy_cr_sample)
            task = self.gen_2d(oxy_task_sample, deoxy_task_sample)
        return task.astype(np.float32), label, 0, oxy_file

    def gen_2d(self, oxy, deoxy):
        data = np.stack([oxy, deoxy], axis=1)
        zero = np.zeros((oxy.shape[0], 2))
        data_0 = data[:,:,0:10]
        data_1 = data[:,:,10:21]
        data_2 = data[:,:,21:31]
        data_3 = data[:,:,31:42]
        data_4 = data[:,:,42:]

        # keep 5x11
        # data_0 = np.stack([data_0[:,:,i] if i != 10 else zero for i in range(11)], axis=2)
        # data_2 = np.stack([data_2[:,:,i] if i != 10 else zero for i in range(11)], axis=2)
        # data_4 = np.stack([data_4[:,:,i] if i != 10 else zero for i in range(11)], axis=2)
        # assert data_0.shape == data_1.shape, "Unpaired shape between odd and even"

        # Adding zeros
        data_0 = np.stack([data_0[:,:,i//2] if i%2 != 0 else zero for i in range(21)], axis=2)
        data_2 = np.stack([data_2[:,:,i//2] if i%2 != 0 else zero for i in range(21)], axis=2)
        data_4 = np.stack([data_4[:,:,i//2] if i%2 != 0 else zero for i in range(21)], axis=2)
    
        data_1 = np.stack([data_1[:,:,i//2] if i%2 == 0 else zero for i in range(21)], axis=2)
        data_3 = np.stack([data_3[:,:,i//2] if i%2 == 0 else zero for i in range(21)], axis=2)


        data = np.stack([data_0, data_1, data_2, data_3, data_4], axis=2)
        return data

    def get_cr_files(self, file):
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

    def collect_data_files_parts(self, lines):
        data_root, _ = os.path.split(lines[0].rstrip())
        for l in lines:
            l = l.rstrip()
            base = os.path.basename(l)
            meta = base.split('_')

            type, id, task, part = meta
            deoxy_file = '_'.join(['deoxy', id, task, part])

            if self.mode in ["train"] and id in self.data_config['ids'] and task in self.data_config["train_tasks"] and part in ["head.npy"]:
                self.data_files.append([l, os.path.join(data_root, deoxy_file)])
            elif self.mode in ["eval"] and id in self.data_config['ids'] and task in self.data_config["eval_tasks"] and part in ["tail.npy"]:
                self.data_files.append([l,os.path.join(data_root, deoxy_file)])
            else:
                continue

    def collect_data_files_trails(self, lines):
        data_root, _ = os.path.split(lines[0].rstrip())
        for l in lines:
            l = l.rstrip()
            base = os.path.basename(l)
            meta = base.split('_')

            type, id, task, part = meta
            deoxy_file = '_'.join(['deoxy', id, task, part])

            if self.mode in ["train"] and id in self.data_config['ids'] and task in self.data_config["train_tasks"] and part in self.data_config["parts"]:
                self.data_files.append([l, os.path.join(data_root, deoxy_file)])
            elif self.mode in ["eval", "test"] and id in self.data_config['ids'] and task in self.data_config["eval_tasks"] and part in self.data_config["parts"]:
                self.data_files.append([l,os.path.join(data_root, deoxy_file)])
            else:
                continue

    def collect_data_files_subjects(self, lines):
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

    def collect_data_files_sessions(self, lines):
        data_root, _ = os.path.split(lines[0].rstrip())
        for l in lines:
            l = l.rstrip()
            base = os.path.basename(l)
            meta = base.split('_')

            type, id, task, part = meta
            deoxy_file = '_'.join(['deoxy', id, task, part])

            if self.mode in ["train"] and id in self.data_config['ids'] and task in self.data_config["train_tasks"] and part in self.data_config["parts"]:
                self.data_files.append([l, os.path.join(data_root, deoxy_file)])
            elif self.mode in ["eval"] and id in self.data_config['ids'] and task in self.data_config["eval_tasks"] and part in self.data_config["parts"]:
                self.data_files.append([l,os.path.join(data_root, deoxy_file)])
            else:
                continue

    def __len__(self) -> int:
        if self.sampling in ['task', 'cr_task']:
            return len(self.instructor)
        return len(self.data_files)