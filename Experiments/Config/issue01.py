import os

class Basic_Config():
    def __init__(self):
        self.seed = 777
        self.device = 'cuda'

        self.list_path = "./Data/FileList/Oxy"
        self.summary_register = ["train"]
        self.summary_dir = "./Experiments/Summary/"
        self.weight_root = "../model_weights/fNIRS_DeeperLook/"