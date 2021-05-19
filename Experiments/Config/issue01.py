import os
import torch
import torch.nn as nn

class Basic_Config():
    def __init__(self):
        self.seed = 777
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
    """
    def __init__(self, mode, logfile):
        super(EXP01, self).__init__()
        self.mode = mode
        self.logfile = logfile
        assert self.mode in ['train', 'eval', 'test', 'debug']

        self.name = self.issue + '_' + self.__class__.__name__
        # data
        self.list_path = "./Data/FileList/ZSCORE_Oxy/"
        self.dataset = "fNIRS_Block"
        self.data_config = {
            # "nb", "anb", "rt", "gng", "ewm", "es"
                    "train_tasks": ["anb" , "rt","ewm", "gng"],
                    "eval_tasks": ["anb" , "rt","ewm", "gng"],
                    "ids":      ["2001", "2004", "2012", "2013", "2015",
                                "8204","8206","8209","8210","8211","8213",
                                "8214","8218", "2006", "2011", "2014", "2017", 
                                "8201", "8203","8208","8216", "2003"],  # no subject named 8012
                    "sessions": ["s1","s2"],
                    "parts": ["head.npy"],
                    "ins_root": "./Data/Ins/label_balance/22Folds/"
        }
        self.steps_sizes = [50, 50, 25]
        self.sampling = "task"
        self.drop_last = True
        self.batch_size = 64
        self.out_mode = 1  # 1 means 2d images as input. 
        self.train_ratio = 0.7

        # save related
        self.summary = False
        self.monitor = 'loss'
        self.save_dir = os.path.join(self.ckpt_root, self.name)

        # train:
        self.resume = False
        self.model = "BiGRU_Attn_Multi_Branch_SLA"
        self.lr = 1e-4
        self.weight_decay = 2e-5
        self.eval_freq = 1
        self.epochs = 100
        self.patience = 5