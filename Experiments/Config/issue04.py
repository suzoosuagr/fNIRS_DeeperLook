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

        self.list_path = "./Data/FileList/finger_tap_ZSCORE_OXY/"
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
        self.list_path = "./Data/FileList/finger_tap_ZSCORE_OXY/"
        self.data_config = {
            # "nb", "anb", "rt", "gng", "ewm", "es"
                    "train_tasks": ['RH', 'LH', 'BH', 'RL', 'LL', 'BL'],
                    "eval_tasks": ['RH', 'LH', 'BH', 'RL', 'LL', 'BL'],
                    "pick" : 19, # the number of maximum pick from each participant.
                    "sessions": ["s1"],
                    "parts": ["head.npy"],
                    "ins_root": "./Data/Ins/finger_tap_label_balance_zscore/5Folds/"
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
        self.lr = 5e-4
        self.weight_decay = 2e-5
        self.eval_freq = 1
        self.epochs = 50
        self.patience = 5


class EXP03(EXP01):
    def __init__(self, mode, logfile):
        super(EXP03, self).__init__(mode, logfile)
        self.model = "CNN1"
        self.lr = 1e-3
        