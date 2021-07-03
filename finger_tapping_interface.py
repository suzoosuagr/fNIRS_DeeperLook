# %% Initialization
import pandas as pd
import numpy as np
from Tools.utils import get_files, msg, ensure
import os
from scipy.signal import butter, lfilter
from scipy.stats import stats

dataRoot = '../data/fNIRS_Data/fNIRS_finger_tapping_data_with_headers/'
saveBigChucksPath = '../data/fNIRS_Data/fNIRS_finger_tapping_big_chunks/'
ensure(saveBigChucksPath)

def butterBandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butterBandpassFilter(data, lowcut, highcut, fs, order=1):
    b, a = butterBandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def zscoreNorm(data):
    return stats.zscore(data, axis=1)


@msg
def matchFiles(dataRoot):
    condiFiles = get_files(dataRoot, extension_filter='.tri')
    datafileHead = 'hb_participant_'
    matchCondiFiles = []
    matchDataFiles = []
    for i in condiFiles:
        basename = os.path.basename(i)
        _, id, _, _ = basename.split('_')
        datafile = os.path.join(dataRoot, datafileHead+f'{id}.csv')
        if os.path.exists(datafile):
            matchDataFiles.append(datafile)
            matchCondiFiles.append(i)
    print(f"Found {len(condiFiles)} condiFiles -> Matched {len(matchCondiFiles)} Files")
    return matchCondiFiles, matchDataFiles

condiFiles, dataFiles = matchFiles(dataRoot)

def redirect_files(condiFiles, redirect):
    path_list = []
    for c in condiFiles:
        path_list.append(os.path.join(redirect, os.path.basename(c)))
    return path_list
    


@msg
def saveDF2CSV(df, path):
    df.to_csv(path, index=False)
    print(f"saved {os.path.basename(path)}")

def withHead2BigChuncks(condiFiles, dataFiles, saveBigChucksPath):
    count = 0
    total = len(dataFiles)
    condiFiles = redirect_files(condiFiles, './temp/')
    for c, d in zip(condiFiles, dataFiles):
        df_d = pd.read_csv(d)
        oxydf = df_d[[f'CH_{i}_Oxy' for i in range(48)]]
        deoxydf = df_d[[f'CH_{i}_DeOxy' for i in range(48)]]
        bandOxy_array = butterBandpassFilter(oxydf.values, 0.01, 0.5, 10.2)
        bandDeOxy_array = butterBandpassFilter(deoxydf.values, 0.01, 0.5, 10.2)
        bandOxy_array = zscoreNorm(bandOxy_array)
        bandDeOxy_array = zscoreNorm(bandDeOxy_array)
        # probe_index = np.arange(len(bandOxy_array)).reshape(-1, 1)+1
        # df_probe_oxy= pd.DataFrame(probe_index, columns=['Probe1(Oxy)'])
        # df_probe_deoxy = pd.DataFrame(probe_index, columns=['Probe1(Oxy)'])
        
        df_oxy = pd.DataFrame(bandOxy_array, index=range(len(bandOxy_array)), columns=[f'CH{i}' for i in range(48)])
        # df_oxy = pd.concat([df_probe_oxy, df_oxy], axis=1)
        df_deoxy = pd.DataFrame(bandDeOxy_array, index=range(len(bandOxy_array)), columns=[f'CH{i}' for i in range(48)])
        # df_deoxy = pd.concat([df_probe_deoxy, df_deoxy], axis=1)
        condiMeta = readCondi(c)
        if condiMeta is None:
            continue
        chuckSlice(condiMeta, df_oxy, df_deoxy, saveBigChucksPath, c)
        count += 1
    print(f"processed {count}/{total}")

def chuckSlice(meta, df_oxy, df_deoxy, save_path, condi_file):
    id = os.path.basename(condi_file).split('/')[-1].split("_")[1]
    for task in meta:
        meta_data = meta[task]
        start, end= meta_data
        new_oxy_file = os.path.join(save_path, f'oxy_{id}_{task}_head.npy')
        new_deoxy_file = os.path.join(save_path, f'deoxy_{id}_{task}_head.npy')

        try:
            data = df_oxy.iloc[start:end].values
            ddata = df_deoxy.iloc[start:end].values
        except ValueError:
            print("ERROR: {id}_{task}->start:{start}, end:{end}")
        assert len(data) == len(ddata)

        np.save(new_oxy_file, data)
        np.save(new_deoxy_file, ddata)

    print(f"Done :) {os.path.basename(condi_file)}")


def readCondi(c):
    label_map = {
        4:'RH', # right tapping 120 
        6:'LH',
        8:'BH', # both tapping 120
        5:'BL', 
        7:'LL',
        9:'BL',
        2:'REST', # start of the rest
        3:'REND',   # end of the rest
        10:'TEND'   # end of the task
    }
    with open(c, 'r') as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]
        length = len(lines)
        num = length // 4
        try:
            assert num * 4 == length
        except AssertionError:
            return None
        condi_meta = {}
        for i in range(num):
            li = lines[i*4:i*4+4]
            cr_start = int(li[0].split(';')[-2])
            cr_end   = int(li[1].split(';')[-2])
            task_start = int(li[2].split(';')[-2])
            task_end = int(li[3].split(';')[-2])
            condi_meta[label_map[int(li[2].split(';')[-1])]] = [task_start, task_end]
            condi_meta['CR'+label_map[int(li[2].split(';')[-1])]] = [cr_start, cr_end]
        return condi_meta



withHead2BigChuncks(condiFiles, dataFiles, saveBigChucksPath)


# %%
# data = dataFiles[0]
# condi = condiFiles[0]

# df_d = pd.read_csv(data)
# # %%
# df_d.head()
# # %%
# # df_d[['CH_0_Oxy', 'CH_1_Oxy']]
# df_d[[f'CH_{i}_Oxy' for i in range(48)]]
# df_filter = butterBandpassFilter(df_d.values, 0.01, 0.5, 10.2)
# # %%  TODO here.
# a = np.array([[1, 1, 2, 3], [2, 4, 5, 6]])
# df_a = pd.DataFrame(a, index=range(len(a)), columns=['Probe1(Oxy)', 'CH0', 'CH1', 'CH2'])
# new_data = df_a[[f'CH{i}' for i in range(3)]].values - 0.5 
# first_col = np.arange(len(new_data)).reshape(-1, 1)+1
# df_first = pd.DataFrame(first_col, columns=['Probe1(Oxy)'])
# # %%
# df_b = pd.DataFrame(new_data, index=range(len(new_data)), columns=['CH0', 'CH1', 'CH2'])
# pd.concat([df_first, df_b], axis=1)
# %%
