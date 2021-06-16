# %%
import matplotlib.pyplot as plt
import numpy as np
import json
from Tools.utils import json2dict, get_files

# %% read records to dict
json_root = './Files/permutation/10folds/'
json_files =  get_files(json_root, extension_filter='.json')
print(f"Found {len(json_files)} records.")
# %%
wml_metrics = {
    'accu': [],
    'precision': [],
    'recall': [],
    'f1': []
}
vpl_metrics = {
    'accu': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for jf in json_files:
    temp_dict = json2dict(jf)
    # for the wml
    wml_metrics['accu'].append(np.mean(temp_dict['wml']['accu']))
    wml_metrics['precision'].append(np.mean(temp_dict['wml']['precision']))
    wml_metrics['recall'].append(np.mean(temp_dict['wml']['recall']))
    wml_metrics['f1'].append(np.mean(temp_dict['wml']['f1']))
    # for the vpl
    vpl_metrics['accu'].append(np.mean(temp_dict['vpl']['accu']))
    vpl_metrics['precision'].append(np.mean(temp_dict['vpl']['precision']))
    vpl_metrics['recall'].append(np.mean(temp_dict['vpl']['recall']))
    vpl_metrics['f1'].append(np.mean(temp_dict['vpl']['f1']))

# %% handle nan 
for k, v in wml_metrics.items():
    wml_metrics[k] = np.nan_to_num(v, nan=0.0)
    print(f"WML- {k} - {np.mean(wml_metrics[k])}")
print("\n")
for k, v in vpl_metrics.items():
    vpl_metrics[k] = np.nan_to_num(v, nan=0.0)
    print(f"WML- {k} - {np.mean(vpl_metrics[k])}")

# %%
