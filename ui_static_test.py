# %%
import matplotlib.pyplot as plt
import numpy as np
import json
from Tools.utils import json2dict, get_files, permutation_p_value

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
    print(f"VPL- {k} - {np.mean(vpl_metrics[k])}")

wml_mean_accu = 0.77533
wml_precision = np.mean([0.83920811, 0.71217233, 0.77079324])
wml_recall = np.mean([0.85363458, 0.64022399, 0.83565137])
wml_f1 = np.mean([0.84635987, 0.67428431, 0.80191302])

vpl_mean_accu = 0.77304
vpl_precision = np.mean([0.76453118, 0.84698118, 0.70519783])
vpl_recall = np.mean([0.83541956, 0.8509332,  0.63625758])
vpl_f1 = np.mean([0.79840496, 0.84895259, 0.66895621] )


# %% wml - accu
# fig, axs = plt.subplots(1,1, plt.tight_layout=True)
# axs[0].hist(wml_mean, bins=15)
permutation_p_value(wml_metrics['accu'], wml_mean_accu)


# %%
