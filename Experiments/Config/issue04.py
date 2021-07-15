import os
import torch
import torch.nn as nn

"""
Multi classes. 
"""

class Basic_Config():
    def __init__(self):
        self.seed = 2333
        self.device = 'cuda'

        self.list_path = "./Data/FileList/fNIRS_Natalie_Zscore_Oxy/"
        self.summary_register = ["train"]
        self.summary_dir = './TFrecords/'
        self.ckpt_root = "../model_weights/fNIRS_DeeperLook/"
        self.issue = __file__.split('/')[-1].split('.')[0].upper()
        self.log_root = "./Log/"

class EXP01(Basic_Config):
    """
    using naive embedding method.
    multi - classification
    self - supervised 50 50 25
    full-size label-balanced data
    10 folds finger tapping. 
    """
    def __init__(self, mode, logfile):
        super(EXP01, self).__init__()
        self.mode = mode
        self.logfile = logfile
        assert self.mode in ['train', 'eval', 'test', 'debug']

        self.name = self.issue + '_' + self.__class__.__name__
        # data
        self.list_path = "./Data/FileList/fNIRS_Natalie_Zscore_Oxy/"
        self.data_config = {
            # "nb", "anb", "rt", "gng", "ewm", "es"
                    "train_tasks": ['RH', 'LH', 'BH', 'RL', 'LL', 'BL'],
                    "eval_tasks": ['RH', 'LH', 'BH', 'RL', 'LL', 'BL'],
                    "pick" : 19, # the number of maximum pick from each participant.
                    "sessions": ["s1"],
                    "parts": ["head.npy"],
                    "ins_root": "./Data/Ins/finger_tap_label_balance_natalie/exp01/11Folds/"
        }
        self.steps_sizes = [50, 50, 25]
        self.drop_last = True
        self.batch_size = 64
        self.out_mode = 1  # 1 means 2d images as input. 

        # save related
        self.summary = False

        # train:
        self.resume = False
        self.model = "BiGRU_Attn_Multi_Branch_SLA"
        self.lr = 1e-3
        self.weight_decay = 2e-5
        self.eval_freq = 1
        self.epochs = 50
        self.patience = 5


class EXP02(EXP01):
    """
        similar to exp01, using a small timewindow and time step
    """
    def __init__(self, mode, logfile):
        super(EXP02, self).__init__(mode, logfile)
        self.data_config = {
            # "nb", "anb", "rt", "gng", "ewm", "es"
                    "train_tasks": ['RH', 'LH', 'BH', 'RL', 'LL', 'BL'],
                    "eval_tasks": ['RH', 'LH', 'BH', 'RL', 'LL', 'BL'],
                    "pick" : 19, # the number of maximum pick from each participant.
                    "sessions": ["s1"],
                    "parts": ["head.npy"],
                    "ins_root": "./Data/Ins/finger_tap_label_balance_natalie/exp02/11Folds/"
        }
        self.steps_sizes = [10, 10, 5]

class EXP03(EXP02):
    def __init__(self, mode, logfile):
        super(EXP03, self).__init__(mode, logfile)
        self.data_config = {
            # "nb", "anb", "rt", "gng", "ewm", "es"
                    "train_tasks": ['RH', 'LH', 'BH', 'RL', 'LL', 'BL'],
                    "eval_tasks": ['RH', 'LH', 'BH', 'RL', 'LL', 'BL'],
                    "pick" : 19, # the number of maximum pick from each participant.
                    "sessions": ["s1"],
                    "parts": ["head.npy"],
                    "ins_root": "./Data/Ins/finger_tap_label_balance_natalie/exp01/11Folds/"
        }
        self.steps_sizes = [50, 50, 5]
        self.model = "BiGRUFingerTap"