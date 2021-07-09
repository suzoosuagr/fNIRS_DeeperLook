# %% 
import scipy.stats as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_sla = pd.read_csv('./Files/conf_interval/sla_data.csv')
y_types = ('Accu', 'Percision', 'Recall', 'F1', 'Accu', 'Percision', 'Recall', 'F1')
# fig, ax = plt.subplots()
A = 0.95
y = [0.6832, 0.6814, 0.6856, 0.68, 0.6852, 0.6849, 0.6875, 0.6823]
# xind = np.arange(4)
data_sla_ndarry = data_sla.values
# yerror = []
for i in range(4):
    data = data_sla_ndarry[:, i]
    low, high = st.t.interval(alpha=A, df=len(data)-1, loc=y[i], scale=st.sem(data))
    print("WML_{}: [{:.04}, {:.04}]".format(y_types[i], low, high))
for i in range(4):
    data = data_sla_ndarry[:, i+4]
    low, high = st.t.interval(alpha=A, df=len(data)-1, loc=y[i+4], scale=st.sem(data))
    print("VPL_{}: [{:.04}, {:.04}]".format(y_types[i+4], low, high))
print("mean data")
print(np.mean(data))
# for i in range(4):
#     data = data_sla_ndarry[:, i]
#     low, high = st.t.interval(alpha=A, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
#     yerror.append([low, high])
# yerror = np.abs(np.array(yerror).T - y)

# ax.errorbar(xind, y, yerr=yerror,  fmt='o', capthick=2)
# ax.set_xticks(xind)
# ax.set_xticklabels(y_types)
# ax.set_xlim([-0.5, 4])
# ax.set_ylim([0, 1.0])

# plt.show()
# %%
# %%