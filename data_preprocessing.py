# %% initialization. 
import os 
import numpy as np
from pandas.core.indexes import base
from Tools.utils import get_files, ensure
import pandas as pd

def msg(func):
    def inner1(*args, **kwargs):
        print("="*20)
        out = func(*args, **kwargs)
        return out
    return inner1

# %% functions.
class DataHandler():
    def __init__(self, nohead_root, c_files_root) -> None:
        self.data_root = "../data/fNIRS_Data/"
        self.nohead_root = nohead_root
        self.cfiles_root = c_files_root

    @staticmethod
    def parseBasename(basename):
        return basename.split('.')[0].split('_')
    
    @msg
    def legiCheck(self, session):
        oxy_files = get_files(self.nohead_root, 'Oxy', '.csv')
        deoxy_files, c_files = self.matchFiles(session, oxy_files)
        count = 0
        print(f"num oxy_files : {len(oxy_files)}")
        print(f"num deoxy_files : {len(deoxy_files)}")
        print(f"num c_files : {len(c_files)}")

        for o, d, c in zip(oxy_files, deoxy_files, c_files):
            try:
                assert os.path.exists(o)
                assert os.path.exists(d)
                assert os.path.exists(c)
            except AssertionError:
                print(f"ERROR: Doxy:{os.path.basename(d)} | Cfile:{os.path.basename(c)}")
                continue
            count += 1
        print(f"Matched {count} | Missed {len(oxy_files) - count}")
        return oxy_files, deoxy_files, c_files

    def matchFiles(self, session, oxy_files):
        deoxy_files = []
        c_files = []
        for o in oxy_files:
            basename = os.path.basename(o)
            id, sess, hbm, probe, blood = self.parseBasename(basename)
            d = os.path.join(self.nohead_root, '_'.join([id, sess, hbm, probe, 'Deoxy'])+'.csv')
            c = os.path.join(self.cfiles_root, '_'.join([id, 'fNIRS', 'conditions', session])+'.csv')
            deoxy_files.append(d)
            c_files.append(c)
        return deoxy_files, c_files

    @staticmethod
    def calZscore(df):
        """
        CALCULATING zscore along the time axis for each CH individually. 
        @params:
            - df: dataFrame of oxy file. 
        """
        cols = list(df.columns)
        cols = cols[:52]
        for col in cols:
            df[col] = np.tanh((df[col] - df[col].mean()) / (df[col].std(ddof=1) + 1e-15))
        return df

    @staticmethod
    def readCondi(condi_file):
        """
            Read the Condi onsets from conditional file
                @outputs:
                    - `base`: basename of the condition file
                    - `metadata`": list of [[onset, dura, label], ...]
        """
        base = os.path.basename(condi_file)
        id = base.split('_')[0]
        df = pd.read_csv(condi_file, index_col=0)
        metadata = []
        for i in range(1,25):
            idx = 'Task{}'.format(i)
            idx_next = 'Task{}'.format(i+1)
            value = df[idx].values
            if value[2] == 'cr':
                next_v = df[idx_next].values
                value[2] = 'cr' + next_v[2]
            metadata.append(value)
        return base, metadata

    @msg
    def noHead2bigChunk(self, session, save_path, oxy_file, deoxy_file, c_file):
        """
            extract big chunks. with head , tail. 
            @params:
            @output:
                - head: the head file
                - tail: the tail file
        """
        # read the data
        ensure(save_path)
        oxy = pd.read_csv(oxy_file, header=None, index_col=0)
        deoxy = pd.read_csv(deoxy_file, header=None, index_col=0)
        # Zscore: global: along the cat(head, tail)
        # oxy = self.calZscore(oxy)
        # deoxy = self.calZscore(deoxy)
        basename, metadata = self.readCondi(c_file)
        if not session in basename.split('_')[-1]:
            return
        
        id = basename.split('_')[0]
        for meta in metadata:
            onset, dura, label = meta[:3]
            new_oxy_file = os.path.join(save_path, f'oxy_{id}_{label}_head.npy')
            new_deoxy_file = os.path.join(save_path, f'deoxy_{id}_{label}_head.npy')

            try:
                data = oxy.iloc[int(onset)-1:int(onset)-1+int(dura), 0:52].values
                ddata = deoxy.iloc[int(onset)-1:int(onset)-1+int(dura), 0:52].values
            except ValueError:
                print("ERROR: {} -> {}, {}, {}".format(basename, onset, dura, label))
            assert len(data) == len(ddata), "length unpaired for oxy and deoxy : {}, label{}".format(basename, label)

            # if repeat, then is tail
            if os.path.isfile(new_oxy_file):
                new_oxy_file = os.path.join(save_path, f'oxy_{id}_{label}_tail.npy')
            np.save(new_oxy_file, data)

            if os.path.isfile(new_deoxy_file):
                new_deoxy_file = os.path.join(save_path, 'deoxy_{}_{}_tail.npy'.format(id, label))
            np.save(new_deoxy_file, ddata)

        print(f"DONE :) {basename}")

# %% args. 
nohead_root = '../data/fNIRS_Data/fNIRS_2_8_no_head/s2_no_head/'
cfiles_root = '../data/fNIRS_Data/fNIRS_2_8_no_head/c_files/'
save_path = '../data/fNIRS_Data/fNIRS_2_8_big_chuncks_sessions/s2/'
# %% rrun 
checker = DataHandler(nohead_root, cfiles_root)
oxy_files, deoxy_files, c_files = checker.legiCheck('s2')
for o, d, c in zip(oxy_files, deoxy_files, c_files):
    checker.noHead2bigChunk('s2', save_path, o, d, c)