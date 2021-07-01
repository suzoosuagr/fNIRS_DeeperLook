# %% Initialization
import pandas as pd
import numpy as np
from Tools.utils import get_files, msg
import os
from scipy.signal import butter, lfilter

dataRoot = '../data/fNIRS_Data/fNIRS_finger_tapping_data_with_headers/'

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

def getNohead(condiFiles, dataFiles):
    for c, d in zip(condiFiles, dataFiles):
        df_d = pd.read_csv(data)
        oxydf = df_d[[f'CH_{i}_Oxy' for i in range(48)]]
        deoxydf = df_d[[f'CH_{i}_DeOxy' for i in range(48)]]
        bandOxy_array = butterBandpassFilter(oxydf.values, 0.01, 0.5, 10.2)
        bandDeOxy_array = butterBandpassFilter(deoxydf.values, 0.01, 0.5, 10.2)
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

# %%
data = dataFiles[0]
condi = condiFiles[0]

df_d = pd.read_csv(data)
# %%
df_d.head()
# %%
# df_d[['CH_0_Oxy', 'CH_1_Oxy']]
df_d[[f'CH_{i}_Oxy' for i in range(48)]]
df_filter = butterBandpassFilter(df_d.values, 0.01, 0.5, 10.2)
# %%  TODO here.
a = np.array([[1, 1, 2, 3], [2, 4, 5, 6]])
df_a = pd.DataFrame(a, index=range(len(a)), columns=['Probe1(Oxy)', 'CH0', 'CH1', 'CH2'])
new_data = df_a[[f'CH{i}' for i in range(3)]].values - 0.5 
first_col = np.arange(len(new_data)).reshape(-1, 1)+1
df_first = pd.DataFrame(first_col, columns=['Probe1(Oxy)'])
# %%
df_b = pd.DataFrame(new_data, index=range(len(new_data)), columns=['CH0', 'CH1', 'CH2'])
pd.concat([df_first, df_b], axis=1)
# %%
