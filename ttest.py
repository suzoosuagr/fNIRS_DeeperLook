# %%
import scipy.stats as st
import os
import numpy as np
import pandas as pd

svm_data = pd.read_csv('./Files/conf_interval/svm_data.csv')
sla_data = pd.read_csv('./Files/conf_interval/sla_data.csv', index_col=False)
cnn_data = pd.read_csv('./Files/conf_interval/cnn_data.csv', index_col=False)
metrics_names = ['accu', 'precision', 'recall', 'f1']
wml_cols = ['wml_'+s for s in metrics_names]
vpl_cols = ['vpl_'+s for s in metrics_names]
cols = wml_cols + vpl_cols
sla_data[wml_cols].head()
# %%
# for colName in cols:
#     cnn_data_col = cnn_data[[colName]].values.flatten()
#     sla_data_col = sla_data[[colName]].values.flatten()

#     statistic, pvalue = st.ttest_rel(sla_data_col, cnn_data_col)
#     print(f"SLA - CNN | {colName} | stats : {statistic} pvalue : {pvalue}")
# print("--"*20)

# for colName in cols:
#     svm_data_col = svm_data[[colName]].values.flatten()
#     sla_data_col = sla_data[[colName]].values.flatten()

#     statistic, pvalue = st.ttest_rel(sla_data_col, svm_data_col)
#     print(f"SLA - SVM | {colName} | stats : {statistic} pvalue : {pvalue}")

cnn_data_col = cnn_data[cols].values.flatten()
sla_data_col = sla_data[cols].values.flatten()
svm_data_col = svm_data[cols].values.flatten()
statistic, pvalue = st.ttest_rel(sla_data_col, cnn_data_col)

print(f"SLA-CNN | t-stat {statistic}    pvalue: {pvalue}")
# %%
