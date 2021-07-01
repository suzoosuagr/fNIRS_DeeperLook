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

        self.list_path = "./Data/FileList/Oxy"
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
    leave one subject out (22 folds)
    normalized
    """
    def __init__(self, mode, logfile):
        super(EXP01, self).__init__()
        self.mode = mode
        self.logfile = logfile
        assert self.mode in ['train', 'eval', 'test', 'debug']

        self.name = self.issue + '_' + self.__class__.__name__
        # data
        self.list_path = "./Data/FileList/ZSCORE_Oxy/"
        self.data_config = {
            "train_tasks": ["anb" , "rt","ewm", "gng"],
            "eval_tasks": ["anb" , "rt","ewm", "gng"],
            "ids":      ["2001", "2004", "2012", "2013", "2015",
                        "8204","8206","8209","8210","8211","8213",
                        "8214","8218", "2006", "2011", "2014", "2017", 
                        "8201", "8203","8208","8216", "2003"],  # no subject named 8012
            "sessions": ["s1","s2"],
            "parts": ["head.npy"],
            "ins_root": "./Data/Ins/label_balance_sub/10Folds/"
        }
        self.steps_sizes = [50, 50, 25]
        self.drop_last = True
